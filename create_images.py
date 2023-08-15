"""
author: @cesarasa
"""
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from PIL import Image, ImageDraw, ImageFont
from chembl_webresource_client.new_client import new_client


def draw_AC_images(smiles_compound_1 = None, 
                   smiles_compound_2 = None, 
                   pK_compound_1 = 1,
                   pK_compound_2 = 1,
                   pChEMBL_compound_1 = None,
                   pChEMBL_compound_2 = None):
    """
    The function takes as input two SMILES strings, and two pK values, and creates an image
    with the two molecules, their pK values, and the Tanimoto similarity between them.

    It is not suggested to avoid providing the SMILES. I think it is better to look for it BEFORE 
    getting into the function, and then provide it as input. However, I have added the option to
    automatically look for the canonical smiles. Also, the canonical SMILES provided by the Chembl
    client was always like the one provided by the data in Ruibo's folder. 
    
    The function works as follows:
    - It takes the SMILES strings and creates two RDKit molecules
    - It creates two images, one for each molecule
    - It puts the two images together, leaving some space in the middle
    - It fills the space in the middle with a white rectangle
    - It computes the Tanimoto similarity between the two molecules
    - It annotates in the middle the Tanimoto similarity
    - It annotates the pK value of each molecule, below the molecule
    - It puts the compound IDs on top
    - It saves the image in the images folder
    """
    if smiles_compound_1 is None or smiles_compound_2 is None:
        from chembl_webresource_client.new_client import new_client
        activities = new_client.activity.filter(molecule_chembl_id__in=(pChEMBL_compound_1, pChEMBL_compound_2))
        smiles_compound_1 = activities[0]['canonical_smiles']
        # activities2 = new_client.activity.filter(molecule_chembl_id__in= pChEMBL_compound_2)
        smiles_compound_2 = activities[1]['canonical_smiles']
    elif pChEMBL_compound_1 is None:
        print('Please provide a ChEMBL ID for compound 1')
    elif pChEMBL_compound_2 is None:
        print('Please provide a ChEMBL ID for compound 2')

    print(smiles_compound_1, smiles_compound_2, pK_compound_1, pK_compound_2, pChEMBL_compound_1, pChEMBL_compound_2)
    mol1 = Chem.MolFromSmiles(smiles_compound_1)
    mol2 = Chem.MolFromSmiles(smiles_compound_2)
    img1 = Draw.MolToImage(mol1, size=(500, 500))
    img2 = Draw.MolToImage(mol2, size=(500, 500))
    # Put both images together, leaving some space in the middle
    img = Image.new('RGB', (1100, 500))
    img.paste(img1, (0, 0))
    img.paste(img2, (600, 0))
    # Fill the space in the middle with a white rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle([(500, 0), (600, 500)], fill=(255, 255, 255))


    # Compute the Tanimoto similarity between mol1 and mol2
    fp1 = AllChem.RDKFingerprint(mol1, 2)
    fp2 = AllChem.RDKFingerprint(mol2, 2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    # print(similarity)
    # Annotate in the middle the Tanimoto similarity
    d = ImageDraw.Draw(img)
    d.text((480, 150), f"Tc ={similarity : .3f} ", 
        fill=(0, 0, 0), 
        font=ImageFont.truetype("times.ttf", 30))
    # Annotate the pChEMBL value of each molecule, below the molecule
    d.text((200, 400), f"pK = {pK_compound_1}", 
        fill=(0, 0, 0), 
        font=ImageFont.truetype("times.ttf", 30))
    d.text((800, 400), f"pK = {pK_compound_2}", 
        fill=(0, 0, 0), 
        font=ImageFont.truetype("times.ttf", 30))

    # Put the compound IDs on top
    if pChEMBL_compound_1:
        d.text((150, 80), f"{pChEMBL_compound_1}", 
            fill=(0, 0, 0), 
            font=ImageFont.truetype("times.ttf", 20))
    
    if pChEMBL_compound_2:
        d.text((750, 80), f"{pChEMBL_compound_2}", 
            fill=(0, 0, 0), 
            font=ImageFont.truetype("times.ttf", 20))
    
    # Saving the image
    img.save(f'images/{pChEMBL_compound_1}_{pChEMBL_compound_2}.png')

    return None

def main(second = 2):
    """
    Code to test the function. 
    """
    df = pd.read_csv('data_cp.csv')

    # first_compound = df['Smiles'][0]

    # Compare the first compound with the rest of the compounds
    # tanimoto_similarity = []
    # for i in range(1, len(df)):
    #     tanimoto_similarity.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(first_compound)), 
    # Chem.RDKFingerprint(Chem.MolFromSmiles(df['Smiles'][i]))))

    # # Build the Dataframe
    # df_tanimoto = pd.DataFrame({'Smiles': df['Smiles'][1:], 'Tanimoto': tanimoto_similarity})

    # # Get only the compounds with a Tanimoto similarity greater than 0.9
    # df_tanimoto = df_tanimoto[df_tanimoto['Tanimoto'] > 0.9]

    smiles_compound_1 = df.Smiles[0]
    smiles_compound_2 = df.Smiles[second]
    pK_compound_1 = df['pChEMBL Value'][0]
    pK_compound_2 = df['pChEMBL Value'][second]
    pChEMBL_compound_1 = df['Compound_ID'][0]
    pChEMBL_compound_2 = df['Compound_ID'][second]

    draw_AC_images(smiles_compound_1, 
                smiles_compound_2, 
                pK_compound_1,
                pK_compound_2,
                pChEMBL_compound_1,
                pChEMBL_compound_2)
    
    print('Image created')
    return None

if __name__ == "__main__":
    main()