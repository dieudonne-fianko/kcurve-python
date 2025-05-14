
# **🔍 kcurve.py - Optimisation du k-Nearest Neighbors (k-NN)**  
Ce module implémente une approche avancée du **k-Nearest Neighbors** (_k-NN_) pour analyser et classifier des **données médicales**. Il optimise le choix de _k_, affine les calculs de distance et intègre des méthodes de visualisation des voisins.

---

## **📌 Fonctionnalités**
✔️ **Prétraitement des données** (_nettoyage, normalisation, encodage des cibles_)  
✔️ **Calcul des distances et similarités** (_distance euclidienne, transformation exponentielle_)  
✔️ **Vote majoritaire et prédiction optimisée**  
✔️ **Optimisation du paramètre k** (_courbes de performance_)  
✔️ **Visualisation des voisins proches** (_utilisation de PCA et graphiques interactifs_)  

---

## **📥 Installation**
1️⃣ **Cloner le dépôt GitHub** :
```bash
git clone https://github.com/dieudonne-fianko/kcurve-python.git
cd kcurve-python
```
2️⃣ **Installer les dépendances** :
```bash
pip install -r requirements.txt
```
3️⃣ **Importer le module dans votre projet** :
```python
import kcurve
```

---

## **🚀 Utilisation**
### **1️⃣ Charger et prétraiter les données**
```python
import kcurve
X, y = kcurve.charger_et_preparer_donnees("data.csv", "diagnosis")
```
💡 **Fonction qui nettoie et encode automatiquement les données médicales.**

### **2️⃣ Exécuter l’algorithme k-NN**
```python
prediction = kcurve.knn_predict(X, y, new_patient_data, k=5)
print(f"Prédiction du diagnostic : {prediction}")
```
💡 **Utilise la distance euclidienne et vote majoritaire pour prédire la classe.**

### **3️⃣ Optimiser le paramètre k**
```python
meilleur_k = kcurve.optimiser_k(X, y)
print(f"Valeur optimale de k : {meilleur_k}")
```
💡 **Trouve le meilleur k en minimisant le taux d’erreur.**

### **4️⃣ Visualiser les voisins d’un patient médical**
```python
kcurve.afficher_voisins(X, y, new_patient_data, k=5)
```
💡 **Affiche un graphique interactif des voisins les plus proches.**

---

## **📊 Résultats et Analyse**
Ce module permet d’obtenir :
✔️ **Une meilleure classification des diagnostics médicaux**  
✔️ **Une optimisation du paramètre k pour des prédictions précises**  
✔️ **Une visualisation intuitive des groupes de patients similaires**  

---

## **📄 Licence**
📜 Ce projet est distribué sous la licence **MIT**. Vous êtes libre de l'utiliser et de le modifier.  

---

## **🔗 Liens et Contributions**
- **GitHub** : [Dieudonné Fianko - kcurve-python](https://github.com/dieudonne-fianko/kcurve-python)  
- **Issues & Bugs** : [_Signaler un problème ici_](https://github.com/dieudonne-fianko/kcurve-python/issues)  
- **Propose une amélioration** : [_Faire une pull request_](https://github.com/dieudonne-fianko/kcurve-python/pulls)  

💡 **Envie d’améliorer ce projet ?** Contribue en proposant du code ou en partageant tes retours ! 🚀  

---

✅ **Ton README.md est prêt !**  
📌 **Ajoute ce fichier à ton dépôt GitHub et fais un commit :**  
```bash
git add README.md
git commit -m "Ajout du README détaillé"
git push origin main
```
