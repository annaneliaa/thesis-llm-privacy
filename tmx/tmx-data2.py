import os
from lxml import etree

def extract_translations_from_tmx(tmx_file, source_lang, target_lang, spanish_file, greek_file):
    print('Extracting translations from', tmx_file)
    # Parse the TMX file
    tree = etree.parse(tmx_file)
    root = tree.getroot()

    tus = root.findall('.//tu')

    # Extract and save Spanish and Greek translations
    for tu in tus:
        spanish_text = None
        greek_text = None
        tuvs = tu.findall('./tuv')
        for tuv in tuvs:
            lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')  # Get the language from the 'lang' attribute with the XML namespace
            seg = tuv.find('./seg')
            if lang == source_lang:
                spanish_text = seg.text.strip() if seg is not None else ''
            elif lang == target_lang:
                greek_text = seg.text.strip() if seg is not None else ''
        
        # Save Spanish text to file
        if spanish_text:
            spanish_file.write(spanish_text + '\n')
        
        # Save Greek text to file
        if greek_text:
            greek_file.write(greek_text + '\n')

# Define directory containing .tmx files
directory = './EMEA'

# Define output directory for extracted texts
output_dir = './EMEA'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Source and target languages
source_lang = 'es'
target_lang = 'el'

# Open files in append mode or create new ones
spanish_file = open(os.path.join(output_dir, directory + '.es'), 'a', encoding='utf-8')
greek_file = open(os.path.join(output_dir, directory + '.el'), 'a', encoding='utf-8')

# Process each .tmx file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.tmx'):
        tmx_file = os.path.join(directory, filename)
        extract_translations_from_tmx(tmx_file, source_lang, target_lang, spanish_file, greek_file)

# Close the files
spanish_file.close()
greek_file.close()

print('Done!')