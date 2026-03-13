import os
import sys
import requests
import zipfile
import shutil
import dendrotweaks as dd
from dendrotweaks.biophys.io import MODFileConverter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reformat_wilmes_mod import reformat

converter = MODFileConverter()

os.makedirs('examples', exist_ok=True)
if not os.listdir('examples/'):
    print("Downloading example data...")
    dd.download_example_data('examples')

os.makedirs('examples/Wilmes_2016', exist_ok=True)

# Download Wilmes_2016 zip file from ModelDB, unzip it and move the contents to the examples folder
url = 'https://modeldb.science/download/187603'
response = requests.get(url)
with open('187603.zip', 'wb') as f:
    f.write(response.content)
with zipfile.ZipFile('187603.zip', 'r') as zip_ref:
    zip_ref.extractall('examples/Wilmes_2016/')
os.remove('187603.zip')

# Make a new folder in Wilmes_2016 folder called biophys, copy mod_files contents to it under the directory mod
# use something other than shutil package to copy the contents
os.makedirs('examples/Wilmes_2016/biophys/mod', exist_ok=True)
shutil.copytree('examples/Wilmes_2016/WilmesEtAl2016/mod_files', 'examples/Wilmes_2016/biophys/mod', dirs_exist_ok=True)

# Reformat MOD files for MODFileConverter compatibility
reformat(
    input_dir='examples/Wilmes_2016/biophys/mod',
    output_dir='examples/Wilmes_2016/biophys/tempmod',
)

# Make a jaxley folder in biophys
os.makedirs('examples/Wilmes_2016/biophys/jaxley', exist_ok=True)

# Use for loop to convert all mod files to python files and save them in jaxley directory
for mod_file in os.listdir('examples/Wilmes_2016/biophys/tempmod'):
    try:
        path_to_mod_file = os.path.join('examples', 'Wilmes_2016', 'biophys', 'tempmod', mod_file)
        path_to_python = os.path.join('examples', 'Wilmes_2016', 'biophys', 'jaxley', mod_file.replace('.mod', '.py'))
        path_to_template = os.path.join('src', 'dendrotweaks', 'biophys', 'default_templates', 'jaxley.py')
        converter.convert(path_to_mod_file, path_to_python, path_to_template)
    except Exception as e:
        print(f"Error converting {mod_file}: {e}")
        continue