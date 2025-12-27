//! Verus Function Extractor using verus_syn
//!
//! Extracts functions with their FULL dependency graph across all files in a crate.
//! Builds a global registry of all types, functions, traits, impls, then resolves
//! transitive dependencies for each target function.

use clap::Parser;
use proc_macro2::TokenStream;
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use verus_syn::visit::{self, Visit};
use verus_syn::{
    Expr, ExprCall, ExprPath, FnMode, ImplItemFn, Item, ItemConst, ItemEnum, ItemFn,
    ItemStruct, ItemTrait, ItemType, Signature, Type, TypePath,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Extract functions from Verus source files")]
struct Args {
    /// Input: either a JSONL file OR a directory of repos
    #[arg(short, long)]
    input: String,

    /// Output JSONL file for extracted functions
    #[arg(short, long)]
    output: String,

    /// Path to verus binary for verification
    #[arg(long)]
    verus_path: Option<String>,

    /// Skip verification step
    #[arg(long)]
    skip_verify: bool,

    /// Mode: "jsonl" for single-file samples, "repo" for cross-file extraction
    #[arg(long, default_value = "jsonl")]
    mode: String,

    /// For repo mode: glob pattern to find Verus files (default: **/*.rs)
    #[arg(long, default_value = "**/*.rs")]
    pattern: String,

    /// Output task entries (Task A/B/C) instead of raw functions
    #[arg(long)]
    task_output: bool,

    /// Secondary output file for task entries (if different from main output)
    #[arg(long)]
    task_file: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InputSample {
    id: String,
    source: String,
    source_file: String,
    full_code: String,
}

#[derive(Debug, Serialize)]
struct ExtractedFunction {
    id: String,
    source: String,
    source_file: String,
    function_name: String,
    function_code: String,
    standalone_code: String,
    full_code: String,
    specs: String,
    dependencies: Vec<String>,
    verified: bool,
    verification_error: Option<String>,
    metadata: serde_json::Value,
}

/// Task entry for the training dataset
#[derive(Debug, Serialize)]
struct TaskEntry {
    id: String,
    task: String, // "task_a", "task_b", "task_c"
    input_text: String,
    target_text: String,
    full_verified_code: String,
    source: String,
    source_file: String,
    verified: bool,
    metadata: TaskMetadata,
}

#[derive(Debug, Serialize)]
struct TaskMetadata {
    original_id: String,
    function_name: String,
    has_requires: bool,
    has_ensures: bool,
    has_invariants: bool,
    has_decreases: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bug_type: Option<String>,
}

/// Represents a collected item from the AST
#[derive(Clone, Debug)]
struct AstItem {
    name: String,
    qualified_name: String, // e.g., "crate::module::TypeName"
    source_file: PathBuf,
    kind: ItemKind,
    tokens: TokenStream,
    references: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq)]
enum ItemKind {
    SpecFn,
    ProofFn,
    ExecFn,
    Struct,
    Enum,
    TypeAlias,
    Trait,
    Const,
    Impl,
    TraitImpl, // impl Trait for Type
}

/// Target function to extract
#[derive(Clone, Debug)]
struct TargetFunction {
    name: String,
    source_file: PathBuf,
    tokens: TokenStream,
    specs: String,
    references: HashSet<String>,
}

/// GLOBAL registry of all items across ALL files in the crate
#[derive(Default)]
struct GlobalRegistry {
    /// All items by simple name (may have multiple with same name from different modules)
    items_by_name: HashMap<String, Vec<AstItem>>,
    /// All items by qualified name (unique)
    items_by_qualified: HashMap<String, AstItem>,
    /// Impl blocks keyed by the type they implement for
    impls_for_type: HashMap<String, Vec<AstItem>>,
    /// Trait impls keyed by "TraitName_TypeName"
    trait_impls: HashMap<String, AstItem>,
    /// Target functions (exec functions with specs)
    targets: Vec<TargetFunction>,
    /// Global use statements (deduplicated)
    use_statements: HashSet<String>,
    /// Current module path during traversal
    current_module: Vec<String>,
    /// Current source file during traversal
    current_file: PathBuf,
}

impl GlobalRegistry {
    fn add_item(&mut self, item: AstItem) {
        // Add by simple name
        self.items_by_name
            .entry(item.name.clone())
            .or_default()
            .push(item.clone());

        // Add by qualified name
        self.items_by_qualified
            .insert(item.qualified_name.clone(), item.clone());

        // Track impls
        if item.kind == ItemKind::Impl {
            // Extract type name from impl
            for ref_name in &item.references {
                self.impls_for_type
                    .entry(ref_name.clone())
                    .or_default()
                    .push(item.clone());
            }
        }
    }

    fn get_item(&self, name: &str) -> Option<&AstItem> {
        // First try qualified name
        if let Some(item) = self.items_by_qualified.get(name) {
            return Some(item);
        }
        // Then try simple name (prefer types over functions)
        if let Some(items) = self.items_by_name.get(name) {
            // Prefer struct/enum/type over functions
            for item in items {
                if matches!(
                    item.kind,
                    ItemKind::Struct | ItemKind::Enum | ItemKind::TypeAlias | ItemKind::Trait
                ) {
                    return Some(item);
                }
            }
            // Fall back to first match
            return items.first();
        }
        None
    }

    fn get_impls_for(&self, type_name: &str) -> Vec<&AstItem> {
        self.impls_for_type
            .get(type_name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    fn qualified_name(&self, name: &str) -> String {
        if self.current_module.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", self.current_module.join("::"), name)
        }
    }
}

/// Visitor to collect all items from the AST
struct ItemCollectorVisitor<'a> {
    registry: &'a mut GlobalRegistry,
}

impl<'a> ItemCollectorVisitor<'a> {
    fn new(registry: &'a mut GlobalRegistry) -> Self {
        Self { registry }
    }
}

impl<'ast> Visit<'ast> for ItemCollectorVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let name = node.sig.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_verus_refs(node);

        let kind = match &node.sig.mode {
            FnMode::Spec(_) | FnMode::SpecChecked(_) => ItemKind::SpecFn,
            FnMode::Proof(_) | FnMode::ProofAxiom(_) => ItemKind::ProofFn,
            FnMode::Default | FnMode::Exec(_) => ItemKind::ExecFn,
        };

        // Add to registry
        self.registry.add_item(AstItem {
            name: name.clone(),
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: kind.clone(),
            tokens: tokens.clone(),
            references: refs.clone(),
        });

        // If exec function with specs, add as target
        if kind == ItemKind::ExecFn {
            let specs = extract_specs_from_sig(&node.sig);
            if !specs.is_empty() {
                self.registry.targets.push(TargetFunction {
                    name,
                    source_file: self.registry.current_file.clone(),
                    tokens,
                    specs,
                    references: refs,
                });
            }
        }

        visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        let name = node.sig.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_verus_impl_refs(node);

        let kind = match &node.sig.mode {
            FnMode::Spec(_) | FnMode::SpecChecked(_) => ItemKind::SpecFn,
            FnMode::Proof(_) | FnMode::ProofAxiom(_) => ItemKind::ProofFn,
            FnMode::Default | FnMode::Exec(_) => ItemKind::ExecFn,
        };

        // Add to registry for dependency lookup
        self.registry.add_item(AstItem {
            name: name.clone(),
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: kind.clone(),
            tokens: tokens.clone(),
            references: refs.clone(),
        });

        // NOTE: Don't add impl methods as targets - they have &self/&mut self
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let name = node.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_struct_refs(node);

        self.registry.add_item(AstItem {
            name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Struct,
            tokens,
            references: refs,
        });
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        let name = node.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_enum_refs(node);

        self.registry.add_item(AstItem {
            name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Enum,
            tokens,
            references: refs,
        });
    }

    fn visit_item_type(&mut self, node: &'ast ItemType) {
        let name = node.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_type_refs(node);

        self.registry.add_item(AstItem {
            name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::TypeAlias,
            tokens,
            references: refs,
        });
    }

    fn visit_item_trait(&mut self, node: &'ast ItemTrait) {
        let name = node.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_trait_refs(node);

        self.registry.add_item(AstItem {
            name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Trait,
            tokens,
            references: refs,
        });
    }

    fn visit_item_const(&mut self, node: &'ast ItemConst) {
        let name = node.ident.to_string();
        let qualified = self.registry.qualified_name(&name);
        let tokens = node.to_token_stream();
        let refs = collect_const_refs(node);

        self.registry.add_item(AstItem {
            name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Const,
            tokens,
            references: refs,
        });
    }

    fn visit_item_impl(&mut self, node: &'ast verus_syn::ItemImpl) {
        let type_name = extract_type_name(&node.self_ty);
        let tokens = node.to_token_stream();

        // Check if this is a trait impl
        let (kind, impl_name) = if let Some((_, trait_path, _)) = &node.trait_ {
            let trait_name = trait_path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            (
                ItemKind::TraitImpl,
                format!("impl_{}_{}", trait_name, type_name),
            )
        } else {
            (ItemKind::Impl, format!("impl_{}", type_name))
        };

        let qualified = self.registry.qualified_name(&impl_name);

        // Collect ALL references from the entire impl block (type, trait, and all methods)
        let refs = collect_impl_refs(node);

        self.registry.add_item(AstItem {
            name: impl_name,
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind,
            tokens,
            references: refs,
        });

        // Visit methods inside impl
        for item in &node.items {
            if let verus_syn::ImplItem::Fn(f) = item {
                self.visit_impl_item_fn(f);
            }
        }
    }

    fn visit_item_use(&mut self, node: &'ast verus_syn::ItemUse) {
        let use_str = node.to_token_stream().to_string();
        self.registry.use_statements.insert(use_str);
    }

    fn visit_item_mod(&mut self, node: &'ast verus_syn::ItemMod) {
        let mod_name = node.ident.to_string();
        self.registry.current_module.push(mod_name);

        if let Some((_, items)) = &node.content {
            for item in items {
                self.visit_item(item);
            }
        }

        self.registry.current_module.pop();
    }

    fn visit_item_macro(&mut self, node: &'ast verus_syn::ItemMacro) {
        let macro_name = node
            .mac
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();

        // Parse verus-like macros
        let is_verus_like = matches!(
            macro_name.as_str(),
            "verus" | "verismo" | "verismo_simple" | "state_machine" | "tokenized_state_machine"
        );

        let tokens = &node.mac.tokens;
        if is_verus_like {
            if let Ok(file) = verus_syn::parse2::<verus_syn::File>(tokens.clone()) {
                for item in &file.items {
                    self.visit_item(item);
                }
            }
        } else {
            // Try parsing any unknown macro
            if let Ok(file) = verus_syn::parse2::<verus_syn::File>(tokens.clone()) {
                for item in &file.items {
                    self.visit_item(item);
                }
            }
        }
    }

    fn visit_item(&mut self, node: &'ast Item) {
        match node {
            Item::Fn(f) => self.visit_item_fn(f),
            Item::Struct(s) => self.visit_item_struct(s),
            Item::Enum(e) => self.visit_item_enum(e),
            Item::Type(t) => self.visit_item_type(t),
            Item::Trait(t) => self.visit_item_trait(t),
            Item::Const(c) => self.visit_item_const(c),
            Item::Impl(i) => self.visit_item_impl(i),
            Item::Use(u) => self.visit_item_use(u),
            Item::Mod(m) => self.visit_item_mod(m),
            Item::Macro(m) => self.visit_item_macro(m),
            _ => {}
        }
    }
}

/// Collect references from a verus function
fn collect_verus_refs(node: &ItemFn) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_fn(node);
    collector.refs
}

fn collect_verus_impl_refs(node: &ImplItemFn) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_impl_item_fn(node);
    collector.refs
}

fn collect_struct_refs(node: &ItemStruct) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_struct(node);
    collector.refs
}

fn collect_enum_refs(node: &ItemEnum) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_enum(node);
    collector.refs
}

fn collect_type_refs(node: &ItemType) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_type(node);
    collector.refs
}

fn collect_trait_refs(node: &ItemTrait) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_trait(node);
    collector.refs
}

fn collect_const_refs(node: &ItemConst) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_const(node);
    collector.refs
}

fn collect_impl_refs(node: &verus_syn::ItemImpl) -> HashSet<String> {
    let mut collector = VerusRefCollector::default();
    collector.visit_item_impl(node);
    collector.refs
}

/// Visitor to collect references from verus AST
#[derive(Default)]
struct VerusRefCollector {
    refs: HashSet<String>,
}

impl<'ast> Visit<'ast> for VerusRefCollector {
    fn visit_expr_path(&mut self, node: &'ast ExprPath) {
        if let Some(last) = node.path.segments.last() {
            let name = last.ident.to_string();
            if !is_builtin(&name) {
                self.refs.insert(name);
            }
        }
        if let Some(first) = node.path.segments.first() {
            let name = first.ident.to_string();
            if !is_builtin(&name) && name != "crate" && name != "self" && name != "super" {
                self.refs.insert(name);
            }
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(p) = &*node.func {
            if let Some(last) = p.path.segments.last() {
                let name = last.ident.to_string();
                if !is_builtin(&name) {
                    self.refs.insert(name);
                }
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_type_path(&mut self, node: &'ast TypePath) {
        if let Some(last) = node.path.segments.last() {
            let name = last.ident.to_string();
            if !is_builtin(&name) {
                self.refs.insert(name);
            }
        }
        // Also check first segment for module-qualified types
        if let Some(first) = node.path.segments.first() {
            let name = first.ident.to_string();
            if !is_builtin(&name) && name != "crate" {
                self.refs.insert(name);
            }
        }
        visit::visit_type_path(self, node);
    }
}

/// Extract type name from Type node
fn extract_type_name(ty: &Type) -> String {
    match ty {
        Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_else(|| "Unknown".to_string()),
        _ => "Unknown".to_string(),
    }
}

/// Extract specs from signature (function-level only)
fn extract_specs_from_sig(sig: &Signature) -> String {
    let mut specs = Vec::new();
    let spec = &sig.spec;

    if let Some(req) = &spec.requires {
        specs.push(format!("requires {}", req.exprs.to_token_stream()));
    }
    if let Some(ens) = &spec.ensures {
        specs.push(format!("ensures {}", ens.exprs.to_token_stream()));
    }
    if let Some(dec) = &spec.decreases {
        specs.push(format!(
            "decreases {}",
            dec.decreases.exprs.to_token_stream()
        ));
    }

    specs.join("\n")
}

/// All proof annotations collected from code
#[derive(Debug, Default, Clone, Serialize)]
struct ProofAnnotations {
    /// Function-level requires
    requires: Vec<String>,
    /// Function-level ensures
    ensures: Vec<String>,
    /// Function-level decreases (for recursion)
    fn_decreases: Vec<String>,
    /// Loop invariants
    invariants: Vec<String>,
    /// Loop decreases
    loop_decreases: Vec<String>,
    /// Assert statements
    asserts: Vec<String>,
}

impl ProofAnnotations {
    fn is_empty(&self) -> bool {
        self.requires.is_empty()
            && self.ensures.is_empty()
            && self.fn_decreases.is_empty()
            && self.invariants.is_empty()
            && self.loop_decreases.is_empty()
    }

    fn has_loop_annotations(&self) -> bool {
        !self.invariants.is_empty() || !self.loop_decreases.is_empty()
    }

    /// Format function-level specs only (for Task B input)
    fn format_function_specs(&self) -> String {
        let mut parts = Vec::new();
        for r in &self.requires {
            parts.push(format!("requires {}", r));
        }
        for e in &self.ensures {
            parts.push(format!("ensures {}", e));
        }
        for d in &self.fn_decreases {
            parts.push(format!("decreases {}", d));
        }
        parts.join("\n")
    }

    /// Format all annotations (for Task A target)
    fn format_all(&self) -> String {
        let mut parts = Vec::new();
        for r in &self.requires {
            parts.push(format!("requires {}", r));
        }
        for e in &self.ensures {
            parts.push(format!("ensures {}", e));
        }
        for d in &self.fn_decreases {
            parts.push(format!("decreases {}", d));
        }
        for inv in &self.invariants {
            parts.push(format!("invariant {}", inv));
        }
        for d in &self.loop_decreases {
            parts.push(format!("loop_decreases {}", d));
        }
        parts.join("\n")
    }
}

/// Visitor to extract ALL proof annotations from a function
struct AnnotationExtractor {
    annotations: ProofAnnotations,
}

impl AnnotationExtractor {
    fn new() -> Self {
        Self {
            annotations: ProofAnnotations::default(),
        }
    }

    fn extract_from_fn(node: &ItemFn) -> ProofAnnotations {
        let mut extractor = Self::new();

        // Extract function-level specs
        let spec = &node.sig.spec;
        if let Some(req) = &spec.requires {
            extractor
                .annotations
                .requires
                .push(req.exprs.to_token_stream().to_string());
        }
        if let Some(ens) = &spec.ensures {
            extractor
                .annotations
                .ensures
                .push(ens.exprs.to_token_stream().to_string());
        }
        if let Some(dec) = &spec.decreases {
            extractor
                .annotations
                .fn_decreases
                .push(dec.decreases.exprs.to_token_stream().to_string());
        }

        // Extract loop annotations from body
        extractor.visit_block(&node.block);

        extractor.annotations
    }
}

impl<'ast> Visit<'ast> for AnnotationExtractor {
    fn visit_expr_while(&mut self, node: &'ast verus_syn::ExprWhile) {
        // Extract loop invariants
        if let Some(inv) = &node.invariant {
            self.annotations
                .invariants
                .push(inv.exprs.to_token_stream().to_string());
        }
        if let Some(inv) = &node.invariant_except_break {
            self.annotations
                .invariants
                .push(format!("except_break {}", inv.exprs.to_token_stream()));
        }
        if let Some(inv) = &node.invariant_ensures {
            self.annotations
                .invariants
                .push(format!("ensures {}", inv.exprs.to_token_stream()));
        }
        // Extract loop decreases
        if let Some(dec) = &node.decreases {
            self.annotations
                .loop_decreases
                .push(dec.exprs.to_token_stream().to_string());
        }

        // Continue visiting
        visit::visit_expr_while(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast verus_syn::ExprLoop) {
        // Extract loop invariants
        if let Some(inv) = &node.invariant {
            self.annotations
                .invariants
                .push(inv.exprs.to_token_stream().to_string());
        }
        if let Some(inv) = &node.invariant_except_break {
            self.annotations
                .invariants
                .push(format!("except_break {}", inv.exprs.to_token_stream()));
        }
        if let Some(inv) = &node.invariant_ensures {
            self.annotations
                .invariants
                .push(format!("ensures {}", inv.exprs.to_token_stream()));
        }
        // Extract loop decreases
        if let Some(dec) = &node.decreases {
            self.annotations
                .loop_decreases
                .push(dec.exprs.to_token_stream().to_string());
        }

        // Continue visiting
        visit::visit_expr_loop(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        // Extract assert statements
        if let Expr::Assert(assert_expr) = node {
            self.annotations
                .asserts
                .push(assert_expr.to_token_stream().to_string());
        }
        visit::visit_expr(self, node);
    }
}

/// Strip all proof annotations from code using regex (AST-based stripping is complex)
/// This uses precise patterns based on AST analysis
fn strip_annotations_from_code(code: &str) -> String {
    use regex::Regex;

    let mut result = code.to_string();

    // Remove proof blocks: proof { ... }
    let proof_block = Regex::new(r"\bproof\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}").unwrap();
    result = proof_block.replace_all(&result, "").to_string();

    // Remove assert statements: assert(...); and assert ... by { ... }
    let assert_paren = Regex::new(r"\bassert\s*\([^;]+\)\s*;").unwrap();
    result = assert_paren.replace_all(&result, "").to_string();
    let assert_by = Regex::new(r"\bassert\s+[^;{]+?\s+by\s*\{[^}]*\}\s*;?").unwrap();
    result = assert_by.replace_all(&result, "").to_string();

    // Remove loop invariants: invariant ... { (up to opening brace)
    let invariant = Regex::new(r"\binvariant\s+[^{}]+?(?=\s*\{)").unwrap();
    result = invariant.replace_all(&result, "").to_string();
    let inv_except = Regex::new(r"\binvariant_except_break\s+[^{}]+?(?=\s*\{)").unwrap();
    result = inv_except.replace_all(&result, "").to_string();
    let inv_ensures = Regex::new(r"\binvariant_ensures\s+[^{}]+?(?=\s*\{)").unwrap();
    result = inv_ensures.replace_all(&result, "").to_string();

    // Remove loop decreases: decreases ... { (when inside loop context)
    // This is tricky - we need to remove decreases that come after invariant or right before {
    let loop_dec = Regex::new(r"(?:invariant[^{}]*?)?\bdecreases\s+[^{}]+?(?=\s*\{)").unwrap();
    result = loop_dec.replace_all(&result, "").to_string();

    // Remove function-level ensures: ensures ... { (right before function body)
    let ensures = Regex::new(r"\bensures\s+[^{}]+?(?=\s*\{)").unwrap();
    result = ensures.replace_all(&result, "").to_string();

    // Remove function-level requires: requires ... (ensures|{)
    let requires = Regex::new(r"\brequires\s+[^{}]+?(?=\s*(?:ensures|\{))").unwrap();
    result = requires.replace_all(&result, "").to_string();

    // Remove function-level decreases (for recursion)
    let fn_dec = Regex::new(r"\bdecreases\s+[^{}]+?(?=\s*\{)").unwrap();
    result = fn_dec.replace_all(&result, "").to_string();

    // Clean up whitespace
    let multi_newline = Regex::new(r"\n\s*\n\s*\n").unwrap();
    result = multi_newline.replace_all(&result, "\n\n").to_string();

    result
}

/// Remove specific annotation type for Task C bug simulation
fn remove_annotation_type(code: &str, bug_type: &str) -> Option<String> {
    use regex::Regex;

    let result = match bug_type {
        "missing_ensures" => {
            let re = Regex::new(r"\bensures\s+[^{}]+?(?=\s*\{)").unwrap();
            re.replace_all(code, "").to_string()
        }
        "missing_requires" => {
            let re = Regex::new(r"\brequires\s+[^{}]+?(?=\s*(?:ensures|\{))").unwrap();
            re.replace_all(code, "").to_string()
        }
        "missing_invariant" => {
            let re = Regex::new(r"\binvariant\s+[^{}]+?(?=\s*\{)").unwrap();
            re.replace_all(code, "").to_string()
        }
        "missing_decreases" => {
            let re = Regex::new(r"\bdecreases\s+[^{}]+?(?=\s*\{)").unwrap();
            re.replace_all(code, "").to_string()
        }
        "missing_assert" => {
            let re1 = Regex::new(r"\bassert\s*\([^;]+\)\s*;").unwrap();
            let re2 = Regex::new(r"\bassert\s+[^;{]+?\s+by\s*\{[^}]*\}\s*;?").unwrap();
            let tmp = re1.replace_all(code, "").to_string();
            re2.replace_all(&tmp, "").to_string()
        }
        _ => return None,
    };

    // Only return if we actually removed something
    if result != code {
        // Clean up whitespace
        let multi_newline = Regex::new(r"\n\s*\n\s*\n").unwrap();
        Some(multi_newline.replace_all(&result, "\n\n").to_string())
    } else {
        None
    }
}

/// Extract function signature for Task B
fn extract_fn_signature(code: &str, func_name: &str) -> String {
    use regex::Regex;

    // Match: pub? fn name(...) -> (...)
    let pattern = format!(
        r"(?:pub\s+)?fn\s+{}\s*\([^)]*\)(?:\s*->\s*\([^)]+\))?",
        regex::escape(func_name)
    );

    if let Ok(re) = Regex::new(&pattern) {
        if let Some(m) = re.find(code) {
            return m.as_str().to_string();
        }
    }

    // Fallback: try to find any exec fn
    let fallback = Regex::new(r"(?:pub\s+)?fn\s+\w+\s*\([^)]*\)(?:\s*->\s*\([^)]+\))?").unwrap();
    fallback
        .find(code)
        .map(|m| m.as_str().to_string())
        .unwrap_or_default()
}

/// Generate Task entries from an extracted function
fn generate_task_entries(
    func: &ExtractedFunction,
    annotations: &ProofAnnotations,
) -> Vec<TaskEntry> {
    let mut entries = Vec::new();
    let code = &func.standalone_code;

    // Task A: Code → Specs (input = code without specs, target = all specs)
    if !annotations.is_empty() {
        let stripped_code = strip_annotations_from_code(code);
        let all_specs = annotations.format_all();

        entries.push(TaskEntry {
            id: format!("task_a_{}", func.id),
            task: "task_a".to_string(),
            input_text: stripped_code,
            target_text: all_specs,
            full_verified_code: code.clone(),
            source: func.source.clone(),
            source_file: func.source_file.clone(),
            verified: func.verified,
            metadata: TaskMetadata {
                original_id: func.id.clone(),
                function_name: func.function_name.clone(),
                has_requires: !annotations.requires.is_empty(),
                has_ensures: !annotations.ensures.is_empty(),
                has_invariants: !annotations.invariants.is_empty(),
                has_decreases: !annotations.fn_decreases.is_empty()
                    || !annotations.loop_decreases.is_empty(),
                bug_type: None,
            },
        });
    }

    // Task B: Specs → Code (input = signature + function specs, target = full code)
    let fn_specs = annotations.format_function_specs();
    if !fn_specs.is_empty() {
        let signature = extract_fn_signature(code, &func.function_name);
        if !signature.is_empty() {
            entries.push(TaskEntry {
                id: format!("task_b_{}", func.id),
                task: "task_b".to_string(),
                input_text: format!("{}\n{}", signature, fn_specs),
                target_text: code.clone(),
                full_verified_code: code.clone(),
                source: func.source.clone(),
                source_file: func.source_file.clone(),
                verified: func.verified,
                metadata: TaskMetadata {
                    original_id: func.id.clone(),
                    function_name: func.function_name.clone(),
                    has_requires: !annotations.requires.is_empty(),
                    has_ensures: !annotations.ensures.is_empty(),
                    has_invariants: !annotations.invariants.is_empty(),
                    has_decreases: !annotations.fn_decreases.is_empty(),
                    bug_type: None,
                },
            });
        }
    }

    // Task C: Repair (multiple bug types)
    let bug_types = [
        ("missing_ensures", !annotations.ensures.is_empty()),
        ("missing_requires", !annotations.requires.is_empty()),
        ("missing_invariant", !annotations.invariants.is_empty()),
        (
            "missing_decreases",
            !annotations.fn_decreases.is_empty() || !annotations.loop_decreases.is_empty(),
        ),
        ("missing_assert", !annotations.asserts.is_empty()),
    ];

    for (bug_type, has_annotation) in bug_types {
        if has_annotation {
            if let Some(buggy_code) = remove_annotation_type(code, bug_type) {
                entries.push(TaskEntry {
                    id: format!("task_c_{}_{}", bug_type, func.id),
                    task: "task_c".to_string(),
                    input_text: buggy_code,
                    target_text: code.clone(),
                    full_verified_code: code.clone(),
                    source: func.source.clone(),
                    source_file: func.source_file.clone(),
                    verified: func.verified,
                    metadata: TaskMetadata {
                        original_id: func.id.clone(),
                        function_name: func.function_name.clone(),
                        has_requires: !annotations.requires.is_empty(),
                        has_ensures: !annotations.ensures.is_empty(),
                        has_invariants: !annotations.invariants.is_empty(),
                        has_decreases: !annotations.fn_decreases.is_empty(),
                        bug_type: Some(bug_type.to_string()),
                    },
                });
            }
        }
    }

    entries
}

/// Check if name is a builtin
fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "bool" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
            | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
            | "int" | "nat" | "char" | "str" | "String"
            | "Vec" | "Option" | "Result" | "Some" | "None" | "Ok" | "Err"
            | "true" | "false" | "self" | "Self" | "crate" | "super"
            | "Seq" | "Set" | "Map" | "Multiset" | "Ptr" | "Ghost" | "Tracked"
            | "old" | "forall" | "exists" | "choose" | "assert" | "assume"
            | "proof" | "spec" | "exec" | "broadcast"
            | "Box" | "Arc" | "Rc" | "Cell" | "RefCell"
            | "len" | "push" | "pop" | "get" | "set" | "insert" | "remove"
            | "view" | "deep_view" | "ext_equal"
            | "FnSpec" | "FnOnce" | "Fn" | "FnMut"
            | "PhantomData" | "Copy" | "Clone" | "Default" | "Debug"
            | "PartialEq" | "Eq" | "PartialOrd" | "Ord" | "Hash"
            | "Send" | "Sync" | "Sized" | "Unpin"
            | "requires" | "ensures" | "decreases" | "invariant"
            | "open" | "closed" | "pub" | "fn" | "let" | "mut" | "const"
    )
}

/// Compute transitive dependencies for a function using the GLOBAL registry
fn compute_dependencies(target: &TargetFunction, registry: &GlobalRegistry) -> Vec<AstItem> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = target.references.iter().cloned().collect();
    let mut deps: Vec<AstItem> = Vec::new();

    // Don't include the target function itself
    visited.insert(target.name.clone());

    while let Some(name) = queue.pop_front() {
        if visited.contains(&name) {
            continue;
        }
        visited.insert(name.clone());

        if let Some(item) = registry.get_item(&name) {
            deps.push(item.clone());

            // Add this item's dependencies to queue
            for ref_name in &item.references {
                if !visited.contains(ref_name) {
                    queue.push_back(ref_name.clone());
                }
            }

            // If it's a struct/enum, also get its impl blocks
            if matches!(item.kind, ItemKind::Struct | ItemKind::Enum) {
                for impl_item in registry.get_impls_for(&name) {
                    if !visited.contains(&impl_item.name) {
                        queue.push_back(impl_item.name.clone());
                    }
                }
            }
        }
    }

    deps
}

/// Rewrite paths to remove module prefixes and flatten to simple names
fn rewrite_paths(tokens: TokenStream) -> TokenStream {
    let token_str = tokens.to_string();

    // Remove crate:: prefix
    let mut result = token_str.replace("crate :: ", "").replace("crate::", "");

    // Remove super:: prefix
    result = result.replace("super :: ", "").replace("super::", "");

    // Remove self:: prefix
    result = result.replace("self :: ", "").replace("self::", "");

    // Pattern: module :: module :: Name -> Name (3+ segments)
    let re_pattern3 = regex::Regex::new(r"\b(\w+)\s*::\s*(\w+)\s*::\s*(\w+)\s*::\s*([A-Z]\w*)").unwrap();
    while re_pattern3.is_match(&result) {
        result = re_pattern3.replace_all(&result, "$4").to_string();
    }

    // Pattern: module :: module :: Name -> Name (2 module segments)
    let re_pattern = regex::Regex::new(r"\b(\w+)\s*::\s*(\w+)\s*::\s*([A-Z]\w*)").unwrap();
    while re_pattern.is_match(&result) {
        result = re_pattern.replace_all(&result, "$3").to_string();
    }

    // Pattern: module :: Name -> Name (type names start with uppercase)
    let re_pattern2 = regex::Regex::new(r"\b([a-z]\w*)\s*::\s*([A-Z]\w*)").unwrap();
    while re_pattern2.is_match(&result) {
        result = re_pattern2.replace_all(&result, "$2").to_string();
    }

    // Pattern: module :: lowercase_name -> lowercase_name (for functions/consts)
    // Be more careful here - only strip obvious module prefixes
    let re_fn_pattern = regex::Regex::new(r"\b(arch|lock|tspec|addr|pgtable|mem|snp|cpu|reg|global|vbox|debug|allocator|richos|boot|secret|registers)\s*::\s*(\w+)").unwrap();
    result = re_fn_pattern.replace_all(&result, "$2").to_string();

    result.parse().unwrap_or(tokens)
}

/// Extract individual items from a use statement
fn extract_use_items(use_str: &str) -> Vec<String> {
    let normalized = use_str.replace(" ", "");
    let mut items = Vec::new();

    // Handle "use path::{a, b, c};" pattern
    if let Some(brace_start) = normalized.find("::{") {
        if let Some(brace_end) = normalized.find('}') {
            let base = &normalized[3..brace_start]; // skip "use"
            let item_list = &normalized[brace_start+3..brace_end];
            for item in item_list.split(',') {
                let item = item.trim();
                if !item.is_empty() && item != "self" {
                    items.push(format!("{}::{}", base, item));
                }
            }
        }
    }
    // Handle "use path::Item;" pattern
    else if let Some(last_sep) = normalized.rfind("::") {
        let item = &normalized[last_sep+2..].trim_end_matches(';');
        if !item.is_empty() && *item != "*" {
            items.push(normalized[3..].trim_end_matches(';').to_string());
        }
    }

    items
}

/// Build standalone code from target function and dependencies
fn build_standalone_code(
    target: &TargetFunction,
    deps: &[AstItem],
    use_statements: &HashSet<String>,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Only use vstd::prelude which covers most needs
    // Additional imports cause duplicate definition errors
    parts.push("use vstd::prelude::*;".to_string());
    parts.push("use core::marker::PhantomData;".to_string());

    // Skip all other use statements - they cause too many conflicts
    let _ = use_statements; // Explicitly ignore

    // Start verus block
    parts.push("\nverus! {\n".to_string());

    // Add common constants
    parts.push("pub spec const MAX: int = i64::MAX as int;".to_string());
    parts.push("pub spec const MIN: int = i64::MIN as int;".to_string());

    // Sort dependencies: types first, then specs, then proofs, then exec
    let mut types: Vec<&AstItem> = Vec::new();
    let mut impls: Vec<&AstItem> = Vec::new();
    let mut specs: Vec<&AstItem> = Vec::new();
    let mut proofs: Vec<&AstItem> = Vec::new();
    let mut execs: Vec<&AstItem> = Vec::new();

    for dep in deps {
        match dep.kind {
            ItemKind::Struct
            | ItemKind::Enum
            | ItemKind::TypeAlias
            | ItemKind::Trait
            | ItemKind::Const => types.push(dep),
            ItemKind::Impl | ItemKind::TraitImpl => impls.push(dep),
            ItemKind::SpecFn => specs.push(dep),
            ItemKind::ProofFn => proofs.push(dep),
            ItemKind::ExecFn => execs.push(dep),
        }
    }

    // Add in order
    for item in types {
        parts.push(rewrite_paths(item.tokens.clone()).to_string());
    }
    for item in impls {
        parts.push(rewrite_paths(item.tokens.clone()).to_string());
    }
    for item in specs {
        parts.push(rewrite_paths(item.tokens.clone()).to_string());
    }
    for item in proofs {
        parts.push(rewrite_paths(item.tokens.clone()).to_string());
    }
    for item in execs {
        parts.push(rewrite_paths(item.tokens.clone()).to_string());
    }

    // Add target function
    parts.push(rewrite_paths(target.tokens.clone()).to_string());

    // Close verus block
    parts.push("\n} // verus!".to_string());

    parts.join("\n")
}

/// Verify code with Verus
fn verify_with_verus(code: &str, verus_path: &str) -> Result<bool, String> {
    let temp_path = "/tmp/verus_verify_temp.rs";
    fs::write(temp_path, code).map_err(|e| e.to_string())?;

    let output = Command::new(verus_path)
        .args(&[temp_path, "--crate-type=lib"])
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(true)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(stderr.to_string())
    }
}

/// Find all Verus files in a directory
fn find_verus_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = glob::glob(&format!("{}/**/*.rs", dir.display())) {
        for entry in entries.flatten() {
            // Skip test files and build artifacts
            let path_str = entry.display().to_string();
            if path_str.contains("/target/")
                || path_str.contains("/.git/")
                || path_str.contains("/build/")
            {
                continue;
            }
            files.push(entry);
        }
    }
    files
}

/// Parse a single file and add to registry
fn parse_file_into_registry(path: &Path, registry: &mut GlobalRegistry) -> bool {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Quick check for verus content
    if !content.contains("verus!") && !content.contains("requires") && !content.contains("ensures")
    {
        return false;
    }

    let file = match verus_syn::parse_file(&content) {
        Ok(f) => f,
        Err(_) => return false,
    };

    registry.current_file = path.to_path_buf();
    registry.current_module.clear();

    let mut visitor = ItemCollectorVisitor::new(registry);
    for item in &file.items {
        visitor.visit_item(item);
    }

    true
}

/// Process in repo mode: parse all files first, then extract
fn process_repo(
    repo_dir: &Path,
    verus_path: Option<&str>,
    skip_verify: bool,
    output_file: &mut fs::File,
    mut task_file: Option<&mut fs::File>,
) -> io::Result<(usize, usize, usize)> {
    let mut registry = GlobalRegistry::default();

    // Phase 1: Parse ALL files to build global registry
    eprintln!("Phase 1: Building global registry from all files...");
    let files = find_verus_files(repo_dir);
    let mut parsed_count = 0;

    for (i, file_path) in files.iter().enumerate() {
        if parse_file_into_registry(file_path, &mut registry) {
            parsed_count += 1;
        }
        if (i + 1) % 100 == 0 {
            eprintln!("  Parsed {}/{} files ({} with Verus)", i + 1, files.len(), parsed_count);
        }
    }

    eprintln!(
        "Registry built: {} items, {} targets",
        registry.items_by_name.len(),
        registry.targets.len()
    );

    // Phase 2: Extract each target with cross-file dependencies
    eprintln!("\nPhase 2: Extracting functions with dependencies...");
    let mut extracted_count = 0;
    let mut verified_count = 0;
    let mut task_count = 0;

    let targets = registry.targets.clone();
    for (i, target) in targets.iter().enumerate() {
        // Compute dependencies from global registry
        let deps = compute_dependencies(target, &registry);
        let dep_names: Vec<String> = deps.iter().map(|d| d.name.clone()).collect();

        // Build standalone code
        let standalone_code = build_standalone_code(target, &deps, &registry.use_statements);

        // Verify
        let (verified, verification_error) = if skip_verify {
            (true, None)
        } else if let Some(path) = verus_path {
            match verify_with_verus(&standalone_code, path) {
                Ok(true) => (true, None),
                Ok(false) => (false, Some("Verification failed".to_string())),
                Err(e) => (false, Some(e)),
            }
        } else {
            (true, None)
        };

        if verified {
            verified_count += 1;
        }

        let fn_id = format!(
            "{:x}",
            md5::compute(format!(
                "{}:{}",
                target.source_file.display(),
                target.name
            ))
        );

        let repo_name = repo_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let extracted = ExtractedFunction {
            id: fn_id[..12].to_string(),
            source: repo_name,
            source_file: target.source_file.display().to_string(),
            function_name: target.name.clone(),
            function_code: target.tokens.to_string(),
            standalone_code: standalone_code.clone(),
            full_code: "".to_string(), // Not needed in repo mode
            specs: target.specs.clone(),
            dependencies: dep_names,
            verified,
            verification_error,
            metadata: serde_json::json!({
                "num_dependencies": deps.len(),
                "cross_file_deps": deps.iter().filter(|d| d.source_file != target.source_file).count(),
            }),
        };

        let json = serde_json::to_string(&extracted)?;
        writeln!(output_file, "{}", json)?;
        extracted_count += 1;

        // Generate task entries if task file is provided and function is verified
        if let Some(tf) = task_file.as_mut() {
            if verified {
                // Create annotations from target specs
                let mut annotations = ProofAnnotations::default();
                for line in target.specs.lines() {
                    if line.starts_with("requires") {
                        annotations.requires.push(line.trim_start_matches("requires").trim().to_string());
                    } else if line.starts_with("ensures") {
                        annotations.ensures.push(line.trim_start_matches("ensures").trim().to_string());
                    } else if line.starts_with("decreases") {
                        annotations.fn_decreases.push(line.trim_start_matches("decreases").trim().to_string());
                    }
                }

                let task_entries = generate_task_entries(&extracted, &annotations);
                for entry in task_entries {
                    let entry_json = serde_json::to_string(&entry)?;
                    writeln!(tf, "{}", entry_json)?;
                    task_count += 1;
                }
            }
        }

        if (i + 1) % 50 == 0 {
            eprintln!(
                "  Extracted {}/{} functions ({} verified)",
                i + 1,
                targets.len(),
                verified_count
            );
        }
    }

    Ok((extracted_count, verified_count, task_count))
}

/// Process JSONL mode (original single-file mode)
fn process_jsonl(
    input_path: &str,
    verus_path: Option<&str>,
    skip_verify: bool,
    output_file: &mut fs::File,
    mut task_file: Option<&mut fs::File>,
) -> io::Result<(usize, usize, usize)> {
    let input_file = fs::File::open(input_path)?;
    let reader = io::BufReader::new(input_file);

    let mut total = 0;
    let mut verified = 0;
    let mut task_count = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let sample: InputSample = match serde_json::from_str(&line) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Build per-file registry
        let mut registry = GlobalRegistry::default();
        registry.current_file = PathBuf::from(&sample.source_file);

        let file = match verus_syn::parse_file(&sample.full_code) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Collect items and extract annotations from parsed AST
        let mut fn_annotations: HashMap<String, ProofAnnotations> = HashMap::new();

        {
            let mut visitor = ItemCollectorVisitor::new(&mut registry);
            for item in &file.items {
                visitor.visit_item(item);

                // Extract annotations from functions using AST
                if let Item::Fn(item_fn) = item {
                    let name = item_fn.sig.ident.to_string();
                    let annotations = AnnotationExtractor::extract_from_fn(item_fn);
                    fn_annotations.insert(name, annotations);
                }
            }
        }

        // Process targets
        let targets = registry.targets.clone();
        for target in &targets {
            let deps = compute_dependencies(target, &registry);
            let standalone_code = build_standalone_code(target, &deps, &registry.use_statements);

            let (is_verified, error) = if skip_verify {
                (true, None)
            } else if let Some(path) = verus_path {
                match verify_with_verus(&standalone_code, path) {
                    Ok(true) => (true, None),
                    Err(e) => (false, Some(e)),
                    _ => (false, Some("Failed".to_string())),
                }
            } else {
                (true, None)
            };

            if is_verified {
                verified += 1;
            }

            let fn_id = format!("{:x}", md5::compute(format!("{}:{}", sample.id, target.name)));

            let extracted = ExtractedFunction {
                id: fn_id[..12].to_string(),
                source: sample.source.clone(),
                source_file: sample.source_file.clone(),
                function_name: target.name.clone(),
                function_code: target.tokens.to_string(),
                standalone_code: standalone_code.clone(),
                full_code: sample.full_code.clone(),
                specs: target.specs.clone(),
                dependencies: deps.iter().map(|d| d.name.clone()).collect(),
                verified: is_verified,
                verification_error: error,
                metadata: serde_json::json!({"original_sample_id": sample.id}),
            };

            let json = serde_json::to_string(&extracted)?;
            writeln!(output_file, "{}", json)?;
            total += 1;

            // Generate task entries if task file is provided and function is verified
            if let Some(tf) = task_file.as_mut() {
                if is_verified {
                    // Get AST-extracted annotations or create from target specs
                    let annotations = fn_annotations
                        .get(&target.name)
                        .cloned()
                        .unwrap_or_else(|| {
                            // Fallback: parse specs from target.specs string
                            let mut ann = ProofAnnotations::default();
                            for line in target.specs.lines() {
                                if line.starts_with("requires") {
                                    ann.requires.push(line.trim_start_matches("requires").trim().to_string());
                                } else if line.starts_with("ensures") {
                                    ann.ensures.push(line.trim_start_matches("ensures").trim().to_string());
                                } else if line.starts_with("decreases") {
                                    ann.fn_decreases.push(line.trim_start_matches("decreases").trim().to_string());
                                }
                            }
                            ann
                        });

                    let task_entries = generate_task_entries(&extracted, &annotations);
                    for entry in task_entries {
                        let entry_json = serde_json::to_string(&entry)?;
                        writeln!(tf, "{}", entry_json)?;
                        task_count += 1;
                    }
                }
            }
        }

        if total % 100 == 0 {
            eprintln!("Progress: {} functions ({} verified)", total, verified);
        }
    }

    Ok((total, verified, task_count))
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut output_file = fs::File::create(&args.output)?;

    // Create task output file if needed
    let mut task_file: Option<fs::File> = if args.task_output {
        let task_path = args
            .task_file
            .clone()
            .unwrap_or_else(|| args.output.replace(".jsonl", "_tasks.jsonl"));
        Some(fs::File::create(task_path)?)
    } else {
        None
    };

    let (total, verified, task_count) = if args.mode == "repo" {
        // Repo mode: process directory with cross-file dependencies
        let repo_path = Path::new(&args.input);
        process_repo(
            repo_path,
            args.verus_path.as_deref(),
            args.skip_verify,
            &mut output_file,
            task_file.as_mut(),
        )?
    } else {
        // JSONL mode: original single-file processing
        process_jsonl(
            &args.input,
            args.verus_path.as_deref(),
            args.skip_verify,
            &mut output_file,
            task_file.as_mut(),
        )?
    };

    eprintln!("\n=== EXTRACTION COMPLETE ===");
    eprintln!("Total functions: {}", total);
    eprintln!(
        "Verified: {} ({:.1}%)",
        verified,
        100.0 * verified as f64 / total.max(1) as f64
    );
    if args.task_output {
        eprintln!("Task entries generated: {}", task_count);
    }

    Ok(())
}
