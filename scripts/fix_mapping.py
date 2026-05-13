import json
import os

# Official Oxford 102 Category Names in order (0 to 101)
flower_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", 
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", 
    "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", 
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", 
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", 
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", 
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", 
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", 
    "bolero orchid", "vincas", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", 
    "mexican sunflower", "rose", "california poppy", "osteospermum", "spring crocus", "bearded iris", 
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose mallow", "cardoon", 
    "freesia", "common tulip", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", 
    "gaura", "geranium", "orange dahlia", "pink-and-yellow dahlia", "cautleya spicata", "japanese anemone", 
    "black-eyed susan", "silverbush", "californian paeony", "spring beauty", "hibiscus", "columbine", 
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "cactus dahlia", "gloxinia", "mallows", 
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", 
    "geranium", "orange dahlia", "pink-and-yellow dahlia", "cautleya spicata", "japanese anemone"
]

def write_mapping():
    mapping_path = "output/label_mapping.json"
    
    # Create dictionary {0: "pink primrose", 1: "hard-leaved pocket orchid", ...}
    label_map = {str(i): name for i, name in enumerate(flower_names)}
    
    with open(mapping_path, "w") as f:
        json.dump(label_map, f, indent=4)
    
    print(f"✅ Master, I have manually overwritten the mapping with {len(label_map)} names.")

if __name__ == "__main__":
    write_mapping()