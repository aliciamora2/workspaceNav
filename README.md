# Semantic Navigation and Intent Alignment for Assistive Robots

This repository contains the implementations associated with the following research papers:

1. **Bridging Perception and User Needs (ECMR 2025):** Focuses on semantic mapping and grounded interpretation using a dual-embeddings approach to connect user requests with real-world environments.
2. **Situated Intent Alignment for Grounded Goal Evaluation (RO-MAN 2026):** Implements situated intent alignment for evaluating grounded goals in assistive robotics.

---

## 📁 Structure & Components

* **`semantic_nav/` (ROS Package):** 
  * Designed to enable a mobile robot to navigate real-world environments autonomously. Upon reaching a target location, it captures an image of the scene in real time and performs object analysis to ground the user's intent.
* **`intentVerification` (Standalone Script):** 
  * Verifies whether a natural language command issued by a user can be successfully executed based on an input/loaded image.

---

## 📄 Citation

If you use this code or work in your research, projects, or teaching, please cite the corresponding papers:

```bibtex
@inproceedings{mora2025bridging,
  title        = {Bridging Perception and User Needs: Semantic Mapping with Grounded Interpretation using a Dual-Embeddings Approach},
  author       = {Mora, Alicia and Barber, Ramon and Garrido, Santiago and Moreno, Luis},
  booktitle    = {2025 European Conference on Mobile Robots (ECMR)},
  pages        = {1--6},
  year         = {2025},
  organization = {IEEE}
}

@inproceedings{11714736,
  author       = {Mora, Alicia and Rauso, Giuseppe and Caccavale, Riccardo and Finzi, Alberto and Barber, Ramon},
  booktitle    = {2026 IEEE 35th International Conference on Robot and Human Interactive Communication (RO-MAN)}, 
  title        = {Situated Intent Alignment for Grounded Goal Evaluation in Assistive Robots}, 
  year         = {2026},
  volume       = {},
  number       = {},
  pages        = {2177-2182},
  doi          = {10.1109/RO-MAN72286.2026.11714736}
}
