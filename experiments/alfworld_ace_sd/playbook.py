"""Playbook utilities vendored from ACE AppWorld (Apache-2.0).

Source: https://github.com/ace-agent/ace-appworld/tree/928e86877d34cd10eaba159606386f93a1765090
  experiments/code/ace/playbook.py  (parse_playbook_line, get_next_global_id,
                                     format_playbook_line, apply_curator_operations,
                                     extract_json_from_text)
  experiments/code/ace/utils.py     (get_section_slug)
Modifications: functions copied unchanged except that the relative import of
get_section_slug is replaced by the inlined copy below. Unused helpers omitted.
"""
import json
import re


def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match new playbook sections
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])


def parse_playbook_line(line):
    """Parse a single playbook line to extract components.

    Supports both formats:
    1) "[id] helpful=X harmful=Y :: content"
    2) "[id] content" (counts default to 0)
    """
    text = line.strip()
    # New/primary format with counts
    pattern_full = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern_full, text)
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    # Fallback simple format without counts
    pattern_simple = r'\[([^\]]+)\]\s*(.*)'
    match2 = re.match(pattern_simple, text)
    if match2:
        return {
            'id': match2.group(1),
            'helpful': 0,
            'harmful': 0,
            'content': match2.group(2).strip(),
            'raw_line': line
        }
    return None

def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format (counts removed)."""
    return f"[{bullet_id}] {content}"

def apply_curator_operations(playbook_text, operations, next_id):
    """
    Apply curator operations to playbook
    
    TODO: Future Operations (not implemented yet)
    - UPDATE: Rewrite existing bullets to be more accurate or comprehensive
    - MERGE: Combine related bullets into stronger ones  
    - CREATE_META: Add high-level strategy sections
    - DELETE: Remove outdated or incorrect bullets (if needed)
    """
    lines = playbook_text.strip().split('\n')
    
    # Build section map
    sections = {}
    current_section = "general"
    section_line_map = {}  # Track which line each section header is on
    # import pdb
    # pdb.set_trace()
    for i, line in enumerate(lines):
        if line.strip().startswith('##'):
            # Extract section name and normalize it
            section_header = line.strip()[2:].strip()
            # Normalize: lowercase, spaces->_, &->and, strip trailing ':'
            normalized = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
            current_section = normalized
            section_line_map[current_section] = i
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))
    
    # Process operations
    bullets_to_add = []
    
    for op in operations:
        op_type = op['type']
        
        # TODO: Future operation types (not implemented yet)
        # elif op_type == 'UPDATE':
        #     bullet_id = op.get('bullet_id', '')
                    #     new_content = op.get('content', '')
            #     bullets_to_update[bullet_id] = new_content
        # elif op_type == 'MERGE':
        #     source_ids = op.get('source_ids', [])
        #     bullets_to_delete.update(source_ids)
        #     # Add merged bullet logic here
        # elif op_type == 'CREATE_META':
        #     section_name = op.get('section_name', 'META_STRATEGIES')
        #     # Add meta section creation logic here
        
        if op_type == 'ADD':
            # Normalize section name from operation
            section_raw = op.get('section', 'general')
            section = section_raw.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
            
            # Check if section exists, if not use 'others'
            if section not in sections and section != 'general':
                print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = 'others'
            
            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1
            
            content = op.get('content', '')
            
            new_line = format_playbook_line(new_id, 0, 0, content)
            bullets_to_add.append((section, new_line))
            print(f"  Added bullet {new_id} to section {section}")
            

    
    # Rebuild playbook
    new_lines = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add new bullets to appropriate sections
    final_lines = []
    current_section = None
    
    for line in new_lines:
        if line.strip().startswith('##'):
            # Before moving to new section, add any bullets for current section
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                # Clear added bullets
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
        final_lines.append(line)
    
    # Add remaining bullets to current section
    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
    
    # If there are still bullets to add (for sections that don't exist), add them to OTHERS
    if bullets_to_add:
        print(f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [b for s, b in bullets_to_add]
        # Find OTHERS section
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        
        if others_idx >= 0:
            # Insert after OTHERS header
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            # Append to end
            final_lines.extend(others_bullets)
    
    return '\n'.join(final_lines), next_id

def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats"""
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: Look for ```json blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == '\\':
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1
            
            return json_objects
        
        # Find all potential JSON objects
        json_objects = find_json_objects(text)
        
        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if text is None:
            print("[WARN] text is None in extract_json_from_text()")
            return None
        else:
            if len(text) > 500:
                print(f"Raw content preview:\n{text[:500]}...")
            else:
                print(f"Raw content:\n{text}")
        
    return None

