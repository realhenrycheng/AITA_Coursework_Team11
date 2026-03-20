from pubmed_loader import PubMedQAData

def build_input(sample, use_mesh=False):
    question = sample["QUESTION"]
    context = " ".join(sample["CONTEXTS"])
    
    text = question + " " + context
    
    if use_mesh:
        mesh = " ".join(sample["MESHES"])
        text += " " + mesh
        
    return text

def get_label(sample):
    return sample["final_decision"]