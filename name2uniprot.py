import requests, json

protlist = ["Retinoblastoma-associated protein", "human papillomavirus", "E1A"]

def name2uniprot(lst):
    uniprotlist = []
    
    for protein_name in lst:
        search_url = "https://rest.uniprot.org/uniprotkb/search?query=(reviewed:true)%20AND%20(protein_name:{0})&fields=accession".format(protein_name)
        response = requests.get(search_url)
        data = json.loads(requests.get(search_url).text)
        try:
            uniprotlist.append([result['primaryAccession'] for result in data['results'] if result['primaryAccession'][0]=='P'][0])
        except:
            uniprotlist.append('')
    return uniprotlist

print(name2uniprot(protlist))