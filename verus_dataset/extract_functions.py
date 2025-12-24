"""
Extract self-contained Verus functions from complex projects.
Creates minimal standalone files that can be verified independently.
"""

import re
import hashlib
import subprocess
import os
from pathlib import Path
from typing import Iterator, Optional
import tempfile

def extract_verus_functions(content: str) -> list[dict]:
    """Extract exec functions with specs from Verus code."""
    functions = []
    
    # Find functions with requires/ensures
    # Pattern: fn name(...) -> ... requires/ensures ... { body }
    pattern = r'((?:pub\s+)?fn\s+(\w+)[^{]*(?:requires|ensures)[^{]*\{)'
    
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for function start with specs
        if re.match(r'\s*(?:pub\s+)?fn\s+\w+', line) and 'spec fn' not in line and 'proof fn' not in line:
            # Check if this function has specs
            func_start = i
            has_specs = False
            brace_count = 0
            func_lines = []
            
            j = i
            while j < len(lines):
                func_lines.append(lines[j])
                if 'requires' in lines[j] or 'ensures' in lines[j]:
                    has_specs = True
                
                brace_count += lines[j].count('{') - lines[j].count('}')
                
                if brace_count > 0 and lines[j].count('}') > 0 and brace_count == 0:
                    # Function ended
                    break
                j += 1
            
            if has_specs and len(func_lines) > 3:
                func_text = '\n'.join(func_lines)
                # Extract function name
                match = re.search(r'fn\s+(\w+)', func_text)
                if match:
                    functions.append({
                        'name': match.group(1),
                        'code': func_text,
                        'start_line': func_start,
                        'end_line': j,
                    })
            
            i = j + 1
        else:
            i += 1
    
    return functions


def create_standalone_file(func_code: str) -> str:
    """Create a minimal standalone Verus file."""
    return f'''use vstd::prelude::*;

verus! {{

{func_code}

}} // verus!
'''


def verify_standalone(code: str, verus_cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Try to verify a standalone code snippet."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        result = subprocess.run(
            [verus_cmd, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0 and "0 errors" in output
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(temp_path)


def extract_and_verify_from_repo(
    repo_path: Path,
    verus_cmd: str,
    max_files: int = 100,
) -> Iterator[dict]:
    """Extract and verify functions from a repo."""
    
    for rs_file in list(repo_path.rglob("*.rs"))[:max_files]:
        try:
            content = rs_file.read_text(encoding='utf-8', errors='replace')
        except:
            continue
        
        # Skip files without verus
        if 'verus!' not in content and 'requires' not in content:
            continue
        
        functions = extract_verus_functions(content)
        
        for func in functions:
            standalone = create_standalone_file(func['code'])
            success, output = verify_standalone(standalone, verus_cmd)
            
            if success:
                yield {
                    'source_file': str(rs_file),
                    'function_name': func['name'],
                    'code': standalone,
                    'original_code': func['code'],
                }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_functions.py <repo_path> <verus_cmd>")
        sys.exit(1)
    
    repo_path = Path(sys.argv[1])
    verus_cmd = sys.argv[2]
    
    count = 0
    for func in extract_and_verify_from_repo(repo_path, verus_cmd):
        print(f"✓ {func['function_name']} from {Path(func['source_file']).name}")
        count += 1
    
    print(f"\nExtracted {count} verifiable functions")
