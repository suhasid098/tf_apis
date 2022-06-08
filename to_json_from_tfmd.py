
import json
from pydoc import doc
import re
import tensorflow as tf
from pathlib import Path


if __name__ == '__main__':
    tf_md = Path('tf.md').read_text('utf-8')

    current_header = ""
    result = []

    for line in tf_md.split('\n'):
        try:
            # Header
            if line.startswith('##'):
                current_header = line.replace('## ', '')

            # Module/class/function
            elif line.startswith('[`'):
                line = line[2:]
                split = line.split('`](')
                name = split[0]
                split = split[1].split(')')
                path = split[0]
                desc = split[1]

                # Normalize name
                name = name.replace('class ', '')
                name = name.replace('(...)', '')

                # Normalize desc
                desc = desc.replace(': ', '')

                docstring = eval(f'tf.{name}.__doc__') or ''
                pattern = re.compile('(>>>.*?\n\n)', re.DOTALL)
                docstring = re.sub(pattern, '```\\1```\n', docstring)

                result.append({'name': name, 'path': path, 'desc': desc, 'type': current_header, 
                    'docs': docstring})
        except (IndexError, TypeError) as e:
            print(f'Error processing line {line} - {e}')
    
    Path(r'C:\Users\suhas\git\tf_frontend\src\meta.json').write_text(json.dumps(result))