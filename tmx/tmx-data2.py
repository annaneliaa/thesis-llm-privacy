import os
from lxml import etree

def extract_translations_from_tmx(tmx_file, source_lang, target_lang, english_file, dutch_file):
    print('Extracting translations from', tmx_file)
    # Parse the TMX file
    tree = etree.parse(tmx_file)
    root = tree.getroot()

    tus = root.findall('.//tu')

    # Extract and save English and Dutch translations
    for tu in tus:
        english_text = None
        dutch_text = None
        tuvs = tu.findall('./tuv')
        for tuv in tuvs:
            lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')  # Get the language from the 'lang' attribute with the XML namespace
            seg = tuv.find('./seg')
            if lang == source_lang:
                english_text = seg.text.strip() if seg is not None else ''
            elif lang == target_lang:
                dutch_text = seg.text.strip() if seg is not None else ''
        
        # Save English text to file
        if english_text:
            english_file.write(english_text + '\n')
        
        # Save Dutch text to file
        if dutch_text:
            dutch_file.write(dutch_text + '\n')

# Define directory containing .tmx files
directory = 'ECDC'

# Define output directory for extracted texts
output_dir = 'ECDC/extracted-texts'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Source and target languages
source_lang = 'en'
target_lang = 'nl'

# Open files in append mode or create new ones
english_file = open(os.path.join(output_dir, directory + '.en'), 'a', encoding='utf-8')
dutch_file = open(os.path.join(output_dir, directory + '.nl'), 'a', encoding='utf-8')

# Process each .tmx file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.tmx'):
        tmx_file = os.path.join(directory, filename)
        extract_translations_from_tmx(tmx_file, source_lang, target_lang, english_file, dutch_file)

# Close the files
english_file.close()
dutch_file.close()

print('Done!')