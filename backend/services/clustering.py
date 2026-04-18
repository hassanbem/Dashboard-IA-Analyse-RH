from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from loguru import logger

class CommentClustering:
    """Clustering des commentaires pour identifier les groupes similaires"""
    custom_stopwords = [
            # Mots génériques de formation
            'formation', 'formateur', 'formatrice', 'session', 'jour', 'jours',
            'cours', 'module', 'participant', 'participants', 'groupe','manque','manqu'
            
            # Adverbes et conjonctions
            'très', 'bien', 'bon', 'bonne', 'mauvais', 'mauvaise', 'merci', 'etc',
            'plus', 'moins', 'faire', 'avoir', 'être', 'peut', 'pouvoir',
            'aussi', 'encore', 'toujours', 'jamais', 'beaucoup', 'peu',
            'assez', 'trop', 'pas', 'sans', 'avec', 'mais', 'donc', 'car',
            
            # Mots vagues
            'correct', 'moyen', 'moyenne', 'rien', 'aucun', 'aucune',
            'ni', 'quelque', 'plusieurs', 'certain', 'certaine',
            
            # Négations et particules
            'ne', 'pas', 'non', 'aucunement', 'nullement',
            
            # Expressions communes
            'de', 'du', 'des', 'un', 'une', 'le', 'la', 'les',
            
            # Mots de liaison
            'pour', 'par', 'sur', 'dans', 'sous', 'vers', 'chez',
            'au', 'aux', 'à', 'en', 'depuis', 'pendant',
            
            # Mots génériques de feedback
            'intéressant', 'intéressante', 'interessant', 'interessante',
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'd',
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car','manuqe'
            'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
            'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
            'à', 'au', 'aux', 'en', 'sur', 'pour', 'par', 'dans',
            'avec', 'sans', 'sous', 'vers', 'chez', 'depuis', 'pendant',
            'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir',
            'pouvoir', 'falloir', 'vouloir', 'devoir', 'croire',
            'très', 'plus', 'moins', 'aussi', 'trop', 'assez', 'peu',
            'bien', 'mal', 'mieux', 'pire', 'beaucoup', 'pas', 'ne',
            'oui', 'non', 'si', 'comment', 'pourquoi', 'quand', 'où',
            'tout', 'tous', 'toute', 'toutes', 'autre', 'même', 'tel',
            'quel', 'quelle', 'quels', 'quelles', 'quelque', 'quelques',
            'chaque', 'certain', 'certaine', 'certains', 'certaines',
            'plusieurs', 'aucun', 'aucune', 'nul', 'nulle','manque','manqu'

        ]
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words=self.custom_stopwords
        )
        self.kmeans = None
        self.labels_ = None
    
    def fit_predict(self, texts: List[str]) -> np.ndarray:
        """Applique le clustering sur les textes"""
        if len(texts) < self.n_clusters:
            logger.warning(f"Not enough texts ({len(texts)}) for {self.n_clusters} clusters")
            return np.zeros(len(texts))
        
        try:
            # Vectorisation TF-IDF
            X = self.vectorizer.fit_transform(texts)
            
            # KMeans clustering
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            
            self.labels_ = self.kmeans.fit_predict(X)
            
            logger.info(f"Clustering completed: {self.n_clusters} clusters")
            return self.labels_
        
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return np.zeros(len(texts))
    
    def get_cluster_keywords(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Extrait les mots-clés de chaque cluster"""
        if self.kmeans is None:
            return {}
        #mots dans les colonnes de matrice tf idf
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            # Obtenir le centroïde du cluster
            centroid = self.kmeans.cluster_centers_[cluster_id]
            
            # Top mots du cluster
            top_indices = centroid.argsort()[-n_words:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords
    
    def analyze_clusters(self, df: pd.DataFrame, text_col: str = 'commentaire') -> pd.DataFrame:
        """Analyse complète des clusters"""
        # Appliquer le clustering
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = self.fit_predict(texts)
        
        # Ajouter les labels au DataFrame
        df['cluster'] = labels
        
        # Statistiques par cluster
        cluster_stats = []
        cluster_keywords = self.get_cluster_keywords()
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_satisfaction': cluster_data['satisfaction'].mean(),
                'avg_contenu': cluster_data['contenu'].mean(),
                'avg_logistique': cluster_data['logistique'].mean(),
                'avg_applicabilite': cluster_data['applicabilite'].mean(),
                'keywords': ', '.join(cluster_keywords.get(cluster_id, [])[:5])
            }
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)


class AnomalyDetector:
    """Détecte les anomalies et signaux faibles"""
    
    def __init__(self):
        self.threshold_low = 2.5  # Score bas
        self.threshold_occurrences = 5  # Nombre minimum d'occurrences
    
    def detect_weak_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Détecte les signaux faibles dans les évaluations"""
        signals = []
        
        # 1. Formations avec scores logistique très bas
        logistique_issues = df[df['logistique'] <= self.threshold_low].groupby('formation_id').agg({
            'logistique': 'mean',
            'evaluation_id': 'count',
            'type_formation': 'first',
            'commentaire': lambda x: self._extract_common_issues(x.tolist())
        }).reset_index()
        
        for _, row in logistique_issues.iterrows():
            if row['evaluation_id'] >= self.threshold_occurrences:
                signals.append({
                    'type': 'critique' if row['logistique'] < 2.0 else 'warning',
                    'formation_id': row['formation_id'],
                    'formation_type': row['type_formation'],
                    'issue': 'Problèmes logistiques récurrents',
                    'occurrences': int(row['evaluation_id']),
                    'avg_score': round(row['logistique'], 2),
                    'details': row['commentaire']
                })
        
        # 2. Formateurs avec satisfaction en baisse
        formateur_issues = df[df['satisfaction'] <= self.threshold_low].groupby('formateur_id').agg({
            'satisfaction': 'mean',
            'evaluation_id': 'count',
            'commentaire': lambda x: self._extract_common_issues(x.tolist())
        }).reset_index()
        
        for _, row in formateur_issues.iterrows():
            if row['evaluation_id'] >= self.threshold_occurrences:
                signals.append({
                    'type': 'warning',
                    'formateur_id': row['formateur_id'],
                    'issue': 'Satisfaction formateur en baisse',
                    'occurrences': int(row['evaluation_id']),
                    'avg_score': round(row['satisfaction'], 2),
                    'details': row['commentaire']
                })
        
        # 3. Applicabilité faible par type de formation
        applicabilite_issues = df[df['applicabilite'] <= self.threshold_low].groupby('type_formation').agg({
            'applicabilite': 'mean',
            'evaluation_id': 'count',
            'commentaire': lambda x: self._extract_common_issues(x.tolist())
        }).reset_index()
        
        for _, row in applicabilite_issues.iterrows():
            if row['evaluation_id'] >= self.threshold_occurrences:
                signals.append({
                    'type': 'warning',
                    'formation_type': row['type_formation'],
                    'issue': 'Applicabilité au poste faible',
                    'occurrences': int(row['evaluation_id']),
                    'avg_score': round(row['applicabilite'], 2),
                    'details': row['commentaire']
                })
        
        logger.info(f"Detected {len(signals)} weak signals")
        return signals
    
    def _extract_common_issues(self, comments: List[str]) -> str:
        """Extrait les problèmes communs des commentaires"""
        # Mots-clés négatifs communs
        negative_keywords = {
             'rapide': 0, 'lent': 0, 'mal': 0, 'mauvais': 0,
            'insuffisant': 0, 'manque': 0, 'problème': 0, 'difficile': 0,
            'compliqué': 0, 'long': 0, 'court': 0, 'salle': 0, 'horaire': 0,
            'retard': 0, 'matériel': 0, 'pause': 0,# Pédagogie & Contenu
    'lent': 0, 'rapide': 0, 'difficile': 0, 'compliqué': 0, 'flou': 0,
    'théorique': 0, 'abstrait': 0, 'superficiel': 0, 'insuffisant': 0, 
    'pauvre': 0, 'ennuyeux': 0, 'monotone': 0, 'répétitif': 0, 'inadapté': 0,
    'dense': 0, 'lourd': 0, 'vague': 0, 'basique': 0, 'scolaire': 0,

    # Formateur & Animation
     'désorganisé': 0, 'froid': 0, 'expéditif': 0,
    'hésitant': 0, 'directif': 0, 'passif': 0, 'absence': 0,
    'explication': 0, 'réponse': 0, 'écoute': 0, 'pédagogie': 0,

    # Logistique & Organisation
    'horaire': 0, 'retard': 0, 'long': 0, 'court': 0, 'durée': 0, 
    'pause': 0, 'repas': 0, 'salle': 0, 'bruit': 0, 'froid': 0, 
    'chaud': 0, 'confort': 0, 'accueil': 0, 'organisation': 0,
    'annulation': 0, 'planning': 0, 'rythme': 0,

    # Technique & Matériel
    'problème': 0, 'matériel': 0, 'connexion': 0, 'micro': 0, 'son': 0,
    'vidéo': 0, 'écran': 0, 'logiciel': 0, 'bug': 0, 'panne': 0,
    'accès': 0, 'lien': 0, 'plateforme': 0, 'ordinateur': 0,

    
        }
        
        for comment in comments:
            if isinstance(comment, str):
                comment_lower = comment.lower()
                for keyword in negative_keywords.keys():
                    if keyword in comment_lower:
                        negative_keywords[keyword] += 1
        
        # Top 3 problèmes
        top_issues = sorted(negative_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
        issues = [issue[0] for issue in top_issues if issue[1] > 0]
        
        return ', '.join(issues) if issues else 'Issues non spécifiés'
    
    def detect_outliers_by_formateur(self, df: pd.DataFrame) -> List[Dict]:
        """Détecte les formateurs avec performances anormales"""
        outliers = []
        
        # Calculer les statistiques globales
        global_mean = df['satisfaction'].mean()
        global_std = df['satisfaction'].std()
        
        # Analyser chaque formateur
        formateur_stats = df.groupby('formateur_id').agg({
            'satisfaction': ['mean', 'count'],
            'formation_id': 'nunique'
        }).reset_index()
        
        formateur_stats.columns = ['formateur_id', 'avg_satisfaction', 'count', 'nb_formations']
        
        for _, row in formateur_stats.iterrows():
            if row['count'] < 5:  # Minimum 5 évaluations
                continue
            
            # Détection d'outlier (z-score)
            z_score = (row['avg_satisfaction'] - global_mean) / global_std
            
            if abs(z_score) > 2:  # Plus de 2 écarts-types
                outliers.append({
                    'formateur_id': row['formateur_id'],
                    'avg_satisfaction': round(row['avg_satisfaction'], 2),
                    'nb_evaluations': int(row['count']),
                    'nb_formations': int(row['nb_formations']),
                    'z_score': round(z_score, 2),
                    'status': 'excellent' if z_score > 0 else 'attention'
                })
        
        return outliers