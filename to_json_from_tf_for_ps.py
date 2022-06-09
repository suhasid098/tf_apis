import json
from pydoc import doc
import re
import tensorflow as tf
from pathlib import Path


if __name__ == '__main__':
    tf_ps = Path('List_of_all_TF_API.md').read_text('utf-8')

    current_header = ""
    result = []

    for line in tf_ps.split('\n'):
        try:           
            # primary symbol surrounded by |
            if line.startswith('|'):
                # Normalize name
                name = name.strip('|')

                # Normalize desc
                desc = desc.replace(': ', '')

                docstring = eval(f'tf.{name}.__doc__') or ''
                pattern = re.compile('(>>>.*?\n\n)', re.DOTALL)
                docstring = re.sub(pattern, '```\\1```\n', docstring)

                result.append({'name': name, 'path': path, 'desc': desc, 'type': current_header, 
                    'docs': docstring})
        except (IndexError, TypeError) as e:
            print(f'Error processing line {line} - {e}')
    
    Path(r'C:\Users\suhas\git\tf_frontend\src\meta_primary_symbol.json').write_text(json.dumps(result))