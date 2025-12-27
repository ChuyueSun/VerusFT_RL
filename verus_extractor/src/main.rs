//! Verus function extractor using verus_syn for AST-based extraction.

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
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    output: String,
    #[arg(long)]
    verus_path: Option<String>,
    #[arg(long)]
    skip_verify: bool,
    #[arg(long, default_value = "jsonl")]
    mode: String,
    #[arg(long, default_value = "**/*.rs")]
    pattern: String,
    #[arg(long)]
    task_output: bool,
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

#[derive(Debug, Serialize)]
struct TaskEntry {
    id: String,
    task: String,
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

#[derive(Clone, Debug)]
struct AstItem {
    name: String,
    qualified_name: String,
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
    TraitImpl,
}

#[derive(Clone, Debug)]
struct TargetFunction {
    name: String,
    source_file: PathBuf,
    tokens: TokenStream,
    specs: String,
    references: HashSet<String>,
}

#[derive(Default)]
struct GlobalRegistry {
    items_by_name: HashMap<String, Vec<AstItem>>,
    items_by_qualified: HashMap<String, AstItem>,
    impls_for_type: HashMap<String, Vec<AstItem>>,
    trait_impls: HashMap<String, AstItem>,
    targets: Vec<TargetFunction>,
    use_statements: HashSet<String>,
    current_module: Vec<String>,
    current_file: PathBuf,
}

impl GlobalRegistry {
    fn add_item(&mut self, item: AstItem) {
        self.items_by_name
            .entry(item.name.clone())
            .or_default()
            .push(item.clone());
        self.items_by_qualified
            .insert(item.qualified_name.clone(), item.clone());

        if item.kind == ItemKind::Impl {
            for ref_name in &item.references {
                self.impls_for_type
                    .entry(ref_name.clone())
                    .or_default()
                    .push(item.clone());
            }
        }
    }

    fn get_item(&self, name: &str) -> Option<&AstItem> {
        if let Some(item) = self.items_by_qualified.get(name) {
            return Some(item);
        }
        if let Some(items) = self.items_by_name.get(name) {
            for item in items {
                if matches!(
                    item.kind,
                    ItemKind::Struct | ItemKind::Enum | ItemKind::TypeAlias | ItemKind::Trait
                ) {
                    return Some(item);
                }
            }
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
        let refs = collect_refs_from_fn(node);

        let kind = match &node.sig.mode {
            FnMode::Spec(_) | FnMode::SpecChecked(_) => ItemKind::SpecFn,
            FnMode::Proof(_) | FnMode::ProofAxiom(_) => ItemKind::ProofFn,
            FnMode::Default | FnMode::Exec(_) => ItemKind::ExecFn,
        };

        self.registry.add_item(AstItem {
            name: name.clone(),
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: kind.clone(),
            tokens: tokens.clone(),
            references: refs.clone(),
        });

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
        let refs = collect_refs_from_impl_fn(node);

        let kind = match &node.sig.mode {
            FnMode::Spec(_) | FnMode::SpecChecked(_) => ItemKind::SpecFn,
            FnMode::Proof(_) | FnMode::ProofAxiom(_) => ItemKind::ProofFn,
            FnMode::Default | FnMode::Exec(_) => ItemKind::ExecFn,
        };

        self.registry.add_item(AstItem {
            name: name.clone(),
            qualified_name: qualified,
            source_file: self.registry.current_file.clone(),
            kind: kind.clone(),
            tokens: tokens.clone(),
            references: refs.clone(),
        });
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let name = node.ident.to_string();
        self.registry.add_item(AstItem {
            name,
            qualified_name: self.registry.qualified_name(&node.ident.to_string()),
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Struct,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        let name = node.ident.to_string();
        self.registry.add_item(AstItem {
            name,
            qualified_name: self.registry.qualified_name(&node.ident.to_string()),
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Enum,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });
    }

    fn visit_item_type(&mut self, node: &'ast ItemType) {
        let name = node.ident.to_string();
        self.registry.add_item(AstItem {
            name,
            qualified_name: self.registry.qualified_name(&node.ident.to_string()),
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::TypeAlias,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });
    }

    fn visit_item_trait(&mut self, node: &'ast ItemTrait) {
        let name = node.ident.to_string();
        self.registry.add_item(AstItem {
            name,
            qualified_name: self.registry.qualified_name(&node.ident.to_string()),
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Trait,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });
    }

    fn visit_item_const(&mut self, node: &'ast ItemConst) {
        let name = node.ident.to_string();
        self.registry.add_item(AstItem {
            name,
            qualified_name: self.registry.qualified_name(&node.ident.to_string()),
            source_file: self.registry.current_file.clone(),
            kind: ItemKind::Const,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });
    }

    fn visit_item_impl(&mut self, node: &'ast verus_syn::ItemImpl) {
        let type_name = extract_type_name(&node.self_ty);
        let (kind, impl_name) = if let Some((_, trait_path, _)) = &node.trait_ {
            let trait_name = trait_path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            (ItemKind::TraitImpl, format!("impl_{}_{}", trait_name, type_name))
        } else {
            (ItemKind::Impl, format!("impl_{}", type_name))
        };

        self.registry.add_item(AstItem {
            name: impl_name.clone(),
            qualified_name: self.registry.qualified_name(&impl_name),
            source_file: self.registry.current_file.clone(),
            kind,
            tokens: node.to_token_stream(),
            references: collect_refs_generic(node),
        });

        for item in &node.items {
            if let verus_syn::ImplItem::Fn(f) = item {
                self.visit_impl_item_fn(f);
            }
        }
    }

    fn visit_item_use(&mut self, node: &'ast verus_syn::ItemUse) {
        self.registry.use_statements.insert(node.to_token_stream().to_string());
    }

    fn visit_item_mod(&mut self, node: &'ast verus_syn::ItemMod) {
        self.registry.current_module.push(node.ident.to_string());
        if let Some((_, items)) = &node.content {
            for item in items {
                self.visit_item(item);
            }
        }
        self.registry.current_module.pop();
    }

    fn visit_item_macro(&mut self, node: &'ast verus_syn::ItemMacro) {
        let macro_name = node.mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        if matches!(macro_name.as_str(), "verus" | "verismo" | "verismo_simple" | "state_machine" | "tokenized_state_machine") {
            if let Ok(file) = verus_syn::parse2::<verus_syn::File>(node.mac.tokens.clone()) {
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

fn collect_refs_from_fn(node: &ItemFn) -> HashSet<String> {
    let mut collector = RefCollector::default();
    collector.visit_item_fn(node);
    collector.refs
}

fn collect_refs_from_impl_fn(node: &ImplItemFn) -> HashSet<String> {
    let mut collector = RefCollector::default();
    collector.visit_impl_item_fn(node);
    collector.refs
}

fn collect_refs_generic<T: ToTokens>(node: &T) -> HashSet<String> {
    let mut collector = RefCollector::default();
    let tokens = node.to_token_stream();
    if let Ok(file) = syn::parse2::<syn::File>(tokens) {
        for item in file.items {
            if let syn::Item::Struct(s) = item {
                for field in s.fields {
                    extract_type_refs(&field.ty, &mut collector.refs);
                }
            }
        }
    }
    collector.refs
}

fn extract_type_refs(ty: &syn::Type, refs: &mut HashSet<String>) {
    if let syn::Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            if !is_builtin(&name) {
                refs.insert(name);
            }
        }
    }
}

#[derive(Default)]
struct RefCollector {
    refs: HashSet<String>,
}

impl<'ast> Visit<'ast> for RefCollector {
    fn visit_expr_path(&mut self, node: &'ast ExprPath) {
        if let Some(last) = node.path.segments.last() {
            let name = last.ident.to_string();
            if !is_builtin(&name) {
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
        visit::visit_type_path(self, node);
    }
}

fn extract_type_name(ty: &Type) -> String {
    match ty {
        Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_else(|| "Unknown".to_string()),
        _ => "Unknown".to_string(),
    }
}

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
        specs.push(format!("decreases {}", dec.decreases.exprs.to_token_stream()));
    }
    specs.join("\n")
}

#[derive(Debug, Default, Clone, Serialize)]
struct ProofAnnotations {
    requires: Vec<String>,
    ensures: Vec<String>,
    fn_decreases: Vec<String>,
    invariants: Vec<String>,
    loop_decreases: Vec<String>,
    asserts: Vec<String>,
}

impl ProofAnnotations {
    fn is_empty(&self) -> bool {
        self.requires.is_empty() && self.ensures.is_empty() && self.fn_decreases.is_empty() && self.invariants.is_empty() && self.loop_decreases.is_empty()
    }

    fn format_function_specs(&self) -> String {
        let mut parts = Vec::new();
        for r in &self.requires { parts.push(format!("requires {}", r)); }
        for e in &self.ensures { parts.push(format!("ensures {}", e)); }
        for d in &self.fn_decreases { parts.push(format!("decreases {}", d)); }
        parts.join("\n")
    }

    fn format_all(&self) -> String {
        let mut parts = Vec::new();
        for r in &self.requires { parts.push(format!("requires {}", r)); }
        for e in &self.ensures { parts.push(format!("ensures {}", e)); }
        for d in &self.fn_decreases { parts.push(format!("decreases {}", d)); }
        for inv in &self.invariants { parts.push(format!("invariant {}", inv)); }
        for d in &self.loop_decreases { parts.push(format!("loop_decreases {}", d)); }
        parts.join("\n")
    }
}

struct AnnotationExtractor {
    annotations: ProofAnnotations,
}

impl AnnotationExtractor {
    fn extract_from_fn(node: &ItemFn) -> ProofAnnotations {
        let mut extractor = Self { annotations: ProofAnnotations::default() };
        let spec = &node.sig.spec;
        if let Some(req) = &spec.requires {
            extractor.annotations.requires.push(req.exprs.to_token_stream().to_string());
        }
        if let Some(ens) = &spec.ensures {
            extractor.annotations.ensures.push(ens.exprs.to_token_stream().to_string());
        }
        if let Some(dec) = &spec.decreases {
            extractor.annotations.fn_decreases.push(dec.decreases.exprs.to_token_stream().to_string());
        }
        extractor.visit_block(&node.block);
        extractor.annotations
    }
}

impl<'ast> Visit<'ast> for AnnotationExtractor {
    fn visit_expr_while(&mut self, node: &'ast verus_syn::ExprWhile) {
        if let Some(inv) = &node.invariant {
            self.annotations.invariants.push(inv.exprs.to_token_stream().to_string());
        }
        if let Some(inv) = &node.invariant_except_break {
            self.annotations.invariants.push(format!("except_break {}", inv.exprs.to_token_stream()));
        }
        if let Some(inv) = &node.invariant_ensures {
            self.annotations.invariants.push(format!("ensures {}", inv.exprs.to_token_stream()));
        }
        if let Some(dec) = &node.decreases {
            self.annotations.loop_decreases.push(dec.exprs.to_token_stream().to_string());
        }
        visit::visit_expr_while(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast verus_syn::ExprLoop) {
        if let Some(inv) = &node.invariant {
            self.annotations.invariants.push(inv.exprs.to_token_stream().to_string());
        }
        if let Some(inv) = &node.invariant_except_break {
            self.annotations.invariants.push(format!("except_break {}", inv.exprs.to_token_stream()));
        }
        if let Some(inv) = &node.invariant_ensures {
            self.annotations.invariants.push(format!("ensures {}", inv.exprs.to_token_stream()));
        }
        if let Some(dec) = &node.decreases {
            self.annotations.loop_decreases.push(dec.exprs.to_token_stream().to_string());
        }
        visit::visit_expr_loop(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        if let Expr::Assert(assert_expr) = node {
            self.annotations.asserts.push(assert_expr.to_token_stream().to_string());
        }
        visit::visit_expr(self, node);
    }
}

fn strip_annotations_from_code(code: &str) -> String {
    use regex::Regex;
    let mut result = code.to_string();

    let patterns = [
        r"\bproof\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        r"\bassert\s*\([^;]+\)\s*;",
        r"\bassert\s+[^;{]+?\s+by\s*\{[^}]*\}\s*;?",
        r"\binvariant\s+[^{}]+?(?=\s*\{)",
        r"\binvariant_except_break\s+[^{}]+?(?=\s*\{)",
        r"\binvariant_ensures\s+[^{}]+?(?=\s*\{)",
        r"(?:invariant[^{}]*?)?\bdecreases\s+[^{}]+?(?=\s*\{)",
        r"\bensures\s+[^{}]+?(?=\s*\{)",
        r"\brequires\s+[^{}]+?(?=\s*(?:ensures|\{))",
    ];

    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            result = re.replace_all(&result, "").to_string();
        }
    }

    if let Ok(re) = Regex::new(r"\n\s*\n\s*\n") {
        result = re.replace_all(&result, "\n\n").to_string();
    }
    result
}

fn remove_annotation_type(code: &str, bug_type: &str) -> Option<String> {
    use regex::Regex;
    let pattern = match bug_type {
        "missing_ensures" => r"\bensures\s+[^{}]+?(?=\s*\{)",
        "missing_requires" => r"\brequires\s+[^{}]+?(?=\s*(?:ensures|\{))",
        "missing_invariant" => r"\binvariant\s+[^{}]+?(?=\s*\{)",
        "missing_decreases" => r"\bdecreases\s+[^{}]+?(?=\s*\{)",
        "missing_assert" => {
            let re1 = Regex::new(r"\bassert\s*\([^;]+\)\s*;").ok()?;
            let re2 = Regex::new(r"\bassert\s+[^;{]+?\s+by\s*\{[^}]*\}\s*;?").ok()?;
            let tmp = re1.replace_all(code, "").to_string();
            let result = re2.replace_all(&tmp, "").to_string();
            return if result != code { Some(result) } else { None };
        }
        _ => return None,
    };

    let re = Regex::new(pattern).ok()?;
    let result = re.replace_all(code, "").to_string();
    if result != code { Some(result) } else { None }
}

fn extract_fn_signature(code: &str, func_name: &str) -> String {
    use regex::Regex;
    let pattern = format!(r"(?:pub\s+)?fn\s+{}\s*\([^)]*\)(?:\s*->\s*\([^)]+\))?", regex::escape(func_name));
    if let Ok(re) = Regex::new(&pattern) {
        if let Some(m) = re.find(code) {
            return m.as_str().to_string();
        }
    }
    String::new()
}

fn generate_task_entries(func: &ExtractedFunction, annotations: &ProofAnnotations) -> Vec<TaskEntry> {
    let mut entries = Vec::new();
    let code = &func.standalone_code;

    if !annotations.is_empty() {
        entries.push(TaskEntry {
            id: format!("task_a_{}", func.id),
            task: "task_a".to_string(),
            input_text: strip_annotations_from_code(code),
            target_text: annotations.format_all(),
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
                has_decreases: !annotations.fn_decreases.is_empty() || !annotations.loop_decreases.is_empty(),
                bug_type: None,
            },
        });
    }

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

    let bug_types = [
        ("missing_ensures", !annotations.ensures.is_empty()),
        ("missing_requires", !annotations.requires.is_empty()),
        ("missing_invariant", !annotations.invariants.is_empty()),
        ("missing_decreases", !annotations.fn_decreases.is_empty() || !annotations.loop_decreases.is_empty()),
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

fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "bool" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
            | "int" | "nat" | "char" | "str" | "String" | "Vec" | "Option" | "Result" | "Some" | "None" | "Ok" | "Err"
            | "true" | "false" | "self" | "Self" | "crate" | "super" | "Seq" | "Set" | "Map" | "Multiset" | "Ptr"
            | "Ghost" | "Tracked" | "old" | "forall" | "exists" | "choose" | "assert" | "assume" | "proof" | "spec"
            | "exec" | "broadcast" | "Box" | "Arc" | "Rc" | "Cell" | "RefCell" | "len" | "push" | "pop" | "get" | "set"
            | "insert" | "remove" | "view" | "deep_view" | "ext_equal" | "FnSpec" | "FnOnce" | "Fn" | "FnMut"
            | "PhantomData" | "Copy" | "Clone" | "Default" | "Debug" | "PartialEq" | "Eq" | "PartialOrd" | "Ord"
            | "Hash" | "Send" | "Sync" | "Sized" | "Unpin" | "requires" | "ensures" | "decreases" | "invariant"
            | "open" | "closed" | "pub" | "fn" | "let" | "mut" | "const"
    )
}

fn compute_dependencies(target: &TargetFunction, registry: &GlobalRegistry) -> Vec<AstItem> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = target.references.iter().cloned().collect();
    let mut deps: Vec<AstItem> = Vec::new();

    visited.insert(target.name.clone());

    while let Some(name) = queue.pop_front() {
        if visited.contains(&name) { continue; }
        visited.insert(name.clone());

        if let Some(item) = registry.get_item(&name) {
            deps.push(item.clone());
            for ref_name in &item.references {
                if !visited.contains(ref_name) {
                    queue.push_back(ref_name.clone());
                }
            }
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

fn rewrite_paths(tokens: TokenStream) -> TokenStream {
    let mut result = tokens.to_string()
        .replace("crate :: ", "").replace("crate::", "")
        .replace("super :: ", "").replace("super::", "")
        .replace("self :: ", "").replace("self::", "");

    let patterns = [
        (r"\b(\w+)\s*::\s*(\w+)\s*::\s*(\w+)\s*::\s*([A-Z]\w*)", "$4"),
        (r"\b(\w+)\s*::\s*(\w+)\s*::\s*([A-Z]\w*)", "$3"),
        (r"\b([a-z]\w*)\s*::\s*([A-Z]\w*)", "$2"),
    ];

    for (pattern, replacement) in &patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            while re.is_match(&result) {
                result = re.replace_all(&result, *replacement).to_string();
            }
        }
    }
    result.parse().unwrap_or(tokens)
}

fn build_standalone_code(target: &TargetFunction, deps: &[AstItem], _use_statements: &HashSet<String>) -> String {
    let mut parts = vec![
        "use vstd::prelude::*;".to_string(),
        "use core::marker::PhantomData;".to_string(),
        "\nverus! {\n".to_string(),
        "pub spec const MAX: int = i64::MAX as int;".to_string(),
        "pub spec const MIN: int = i64::MIN as int;".to_string(),
    ];

    let (mut types, mut impls, mut specs, mut proofs, mut execs) = (vec![], vec![], vec![], vec![], vec![]);
    for dep in deps {
        match dep.kind {
            ItemKind::Struct | ItemKind::Enum | ItemKind::TypeAlias | ItemKind::Trait | ItemKind::Const => types.push(dep),
            ItemKind::Impl | ItemKind::TraitImpl => impls.push(dep),
            ItemKind::SpecFn => specs.push(dep),
            ItemKind::ProofFn => proofs.push(dep),
            ItemKind::ExecFn => execs.push(dep),
        }
    }

    for items in [types, impls, specs, proofs, execs] {
        for item in items {
            parts.push(rewrite_paths(item.tokens.clone()).to_string());
        }
    }

    parts.push(rewrite_paths(target.tokens.clone()).to_string());
    parts.push("\n} // verus!".to_string());
    parts.join("\n")
}

fn verify_with_verus(code: &str, verus_path: &str) -> Result<bool, String> {
    let temp_path = "/tmp/verus_verify_temp.rs";
    fs::write(temp_path, code).map_err(|e| e.to_string())?;
    let output = Command::new(verus_path).args([temp_path, "--crate-type=lib"]).output().map_err(|e| e.to_string())?;
    if output.status.success() { Ok(true) } else { Err(String::from_utf8_lossy(&output.stderr).to_string()) }
}

fn find_verus_files(dir: &Path) -> Vec<PathBuf> {
    glob::glob(&format!("{}/**/*.rs", dir.display()))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|p| {
            let s = p.display().to_string();
            !s.contains("/target/") && !s.contains("/.git/") && !s.contains("/build/")
        })
        .collect()
}

fn parse_file_into_registry(path: &Path, registry: &mut GlobalRegistry) -> bool {
    let content = match fs::read_to_string(path) { Ok(c) => c, Err(_) => return false };
    if !content.contains("verus!") && !content.contains("requires") && !content.contains("ensures") { return false; }
    let file = match verus_syn::parse_file(&content) { Ok(f) => f, Err(_) => return false };

    registry.current_file = path.to_path_buf();
    registry.current_module.clear();
    let mut visitor = ItemCollectorVisitor::new(registry);
    for item in &file.items { visitor.visit_item(item); }
    true
}

fn process_repo(repo_dir: &Path, verus_path: Option<&str>, skip_verify: bool, output_file: &mut fs::File, mut task_file: Option<&mut fs::File>) -> io::Result<(usize, usize, usize)> {
    let mut registry = GlobalRegistry::default();
    eprintln!("Phase 1: Building global registry...");

    let files = find_verus_files(repo_dir);
    let mut parsed = 0;
    for (i, path) in files.iter().enumerate() {
        if parse_file_into_registry(path, &mut registry) { parsed += 1; }
        if (i + 1) % 100 == 0 { eprintln!("  Parsed {}/{} files", i + 1, files.len()); }
    }
    eprintln!("Registry: {} items, {} targets", registry.items_by_name.len(), registry.targets.len());

    eprintln!("\nPhase 2: Extracting...");
    let (mut extracted, mut verified, mut tasks) = (0, 0, 0);
    let targets = registry.targets.clone();

    for (i, target) in targets.iter().enumerate() {
        let deps = compute_dependencies(target, &registry);
        let standalone = build_standalone_code(target, &deps, &registry.use_statements);
        let (is_verified, error) = if skip_verify { (true, None) }
        else if let Some(path) = verus_path {
            match verify_with_verus(&standalone, path) { Ok(true) => (true, None), Err(e) => (false, Some(e)), _ => (false, Some("Failed".into())) }
        } else { (true, None) };

        if is_verified { verified += 1; }
        let fn_id = format!("{:x}", md5::compute(format!("{}:{}", target.source_file.display(), target.name)));
        let repo_name = repo_dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "unknown".into());

        let func = ExtractedFunction {
            id: fn_id[..12].to_string(), source: repo_name, source_file: target.source_file.display().to_string(),
            function_name: target.name.clone(), function_code: target.tokens.to_string(), standalone_code: standalone.clone(),
            full_code: String::new(), specs: target.specs.clone(), dependencies: deps.iter().map(|d| d.name.clone()).collect(),
            verified: is_verified, verification_error: error, metadata: serde_json::json!({"num_deps": deps.len()}),
        };

        writeln!(output_file, "{}", serde_json::to_string(&func)?)?;
        extracted += 1;

        if let Some(tf) = task_file.as_mut() {
            if is_verified {
                let mut ann = ProofAnnotations::default();
                for line in target.specs.lines() {
                    if line.starts_with("requires") { ann.requires.push(line.trim_start_matches("requires").trim().into()); }
                    else if line.starts_with("ensures") { ann.ensures.push(line.trim_start_matches("ensures").trim().into()); }
                    else if line.starts_with("decreases") { ann.fn_decreases.push(line.trim_start_matches("decreases").trim().into()); }
                }
                for entry in generate_task_entries(&func, &ann) {
                    writeln!(tf, "{}", serde_json::to_string(&entry)?)?;
                    tasks += 1;
                }
            }
        }
        if (i + 1) % 50 == 0 { eprintln!("  Extracted {}/{} ({} verified)", i + 1, targets.len(), verified); }
    }
    Ok((extracted, verified, tasks))
}

fn process_jsonl(input_path: &str, verus_path: Option<&str>, skip_verify: bool, output_file: &mut fs::File, mut task_file: Option<&mut fs::File>) -> io::Result<(usize, usize, usize)> {
    let reader = io::BufReader::new(fs::File::open(input_path)?);
    let (mut total, mut verified, mut tasks) = (0, 0, 0);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let sample: InputSample = match serde_json::from_str(&line) { Ok(s) => s, Err(_) => continue };

        let mut registry = GlobalRegistry::default();
        registry.current_file = PathBuf::from(&sample.source_file);
        let file = match verus_syn::parse_file(&sample.full_code) { Ok(f) => f, Err(_) => continue };

        let mut fn_annotations: HashMap<String, ProofAnnotations> = HashMap::new();
        {
            let mut visitor = ItemCollectorVisitor::new(&mut registry);
            for item in &file.items {
                visitor.visit_item(item);
                if let Item::Fn(item_fn) = item {
                    fn_annotations.insert(item_fn.sig.ident.to_string(), AnnotationExtractor::extract_from_fn(item_fn));
                }
            }
        }

        let targets = registry.targets.clone();
        for target in &targets {
            let deps = compute_dependencies(target, &registry);
            let standalone = build_standalone_code(target, &deps, &registry.use_statements);
            let (is_verified, error) = if skip_verify { (true, None) }
            else if let Some(path) = verus_path {
                match verify_with_verus(&standalone, path) { Ok(true) => (true, None), Err(e) => (false, Some(e)), _ => (false, Some("Failed".into())) }
            } else { (true, None) };

            if is_verified { verified += 1; }
            let fn_id = format!("{:x}", md5::compute(format!("{}:{}", sample.id, target.name)));

            let func = ExtractedFunction {
                id: fn_id[..12].to_string(), source: sample.source.clone(), source_file: sample.source_file.clone(),
                function_name: target.name.clone(), function_code: target.tokens.to_string(), standalone_code: standalone.clone(),
                full_code: sample.full_code.clone(), specs: target.specs.clone(), dependencies: deps.iter().map(|d| d.name.clone()).collect(),
                verified: is_verified, verification_error: error, metadata: serde_json::json!({"original_sample_id": sample.id}),
            };

            writeln!(output_file, "{}", serde_json::to_string(&func)?)?;
            total += 1;

            if let Some(tf) = task_file.as_mut() {
                if is_verified {
                    let ann = fn_annotations.get(&target.name).cloned().unwrap_or_default();
                    for entry in generate_task_entries(&func, &ann) {
                        writeln!(tf, "{}", serde_json::to_string(&entry)?)?;
                        tasks += 1;
                    }
                }
            }
        }
        if total % 100 == 0 { eprintln!("Progress: {} functions ({} verified)", total, verified); }
    }
    Ok((total, verified, tasks))
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut output_file = fs::File::create(&args.output)?;
    let mut task_file = if args.task_output {
        Some(fs::File::create(args.task_file.clone().unwrap_or_else(|| args.output.replace(".jsonl", "_tasks.jsonl")))?)
    } else { None };

    let (total, verified, tasks) = if args.mode == "repo" {
        process_repo(Path::new(&args.input), args.verus_path.as_deref(), args.skip_verify, &mut output_file, task_file.as_mut())?
    } else {
        process_jsonl(&args.input, args.verus_path.as_deref(), args.skip_verify, &mut output_file, task_file.as_mut())?
    };

    eprintln!("\n=== COMPLETE ===\nTotal: {}\nVerified: {} ({:.1}%)", total, verified, 100.0 * verified as f64 / total.max(1) as f64);
    if args.task_output { eprintln!("Tasks: {}", tasks); }
    Ok(())
}
