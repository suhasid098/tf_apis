import json
import tensorflow as tf
from pathlib import Path


if __name__ == '__main__':
    #Get list of tf primary symbol apis from csv
    tf_ps = Path('List_of_all_TF_API.csv').read_text('utf-8').split('\n')[1:]

    result = []

    for api in tf_ps:

        api = api.strip()
        # Verify the api is not empty
        if api == '':
            continue

        try:
            # Call __doc__ on each api to obtain the docstring
            docs = eval(f'{api}.__doc__') or ''
            # Use the first line of the docstring as the short description
            desc = docs.split('\n')[0]

            # Add to results
            result.append({'name': api, 'docs': docs, 'desc': desc, 'type': 'API'})

        except AttributeError as e:
            print(f'Error processing {api}: {e}')
    Path(r'C:\Users\suhas\git\tf_frontend\src\meta_primary_symbol.json').write_text(json.dumps(result))