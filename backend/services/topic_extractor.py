import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re
from typing import List, Dict, Tuple
import subprocess
import sys
import collections


class TopicExtractor:
    """Extrait les thèmes principaux des commentaires"""
    
    def __init__(self, language: str = "fr"):
        print("🔍 Loading spaCy model for topic extraction...")
        self.nlp = None
        
        try:
            if language == "fr":
                self.nlp = spacy.load("fr_core_news_sm")
                print("✅ spaCy French model loaded successfully")
        except OSError:
            print("⚠️  spaCy model not found, downloading...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "fr_core_news_sm"
                ])
                self.nlp = spacy.load("fr_core_news_sm")
                print("✅ spaCy model downloaded and loaded")
            except Exception as e:
                print(f"❌ Could not download spaCy model: {e}")
                print("⚠️  Topic extraction will use basic text processing")
                self.nlp = None
        
        # ✅ LISTE ÉTENDUE DE STOP WORDS PERSONNALISÉS
        self.custom_stopwords = {
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
            'intéressant', 'intéressante', 'interessant', 'interessante'
        }
        
        # ✅ EXPRESSIONS À SUPPRIMER (bigrammes/trigrammes non informatifs)
        self.stopword_phrases = {
            'pas assez', 'trop de', 'manque de', 'assez rien',
            'ni ni', 'sans sans', 'mais mais', 'correct correct'
        }
    
    def preprocess_for_topics(self, texts: List[str]) -> List[str]:
        """Prétraite les textes pour l'extraction de thèmes"""
        processed = []
        
        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            
            if self.nlp:
                # Analyse avec spaCy
                doc = self.nlp(text.lower())
                
                # ✅ FILTRAGE AMÉLIORÉ
                tokens = []
                for token in doc:
                    # Conditions strictes pour garder un token
                    if (not token.is_stop 
                        and not token.is_punct 
                        and not token.is_space
                        and token.pos_ in ['NOUN', 'ADJ', 'VERB']  # Seulement noms, adjectifs, verbes
                        and len(token.text) > 3  # Minimum 4 caractères
                        and token.lemma_ not in self.custom_stopwords
                        and token.text not in self.custom_stopwords
                        and not token.is_digit  # Pas de chiffres
                        and token.text.isalpha()):  # Seulement des lettres
                        
                        tokens.append(token.lemma_)
            else:
                # Fallback : traitement basique
                tokens = self._basic_tokenize(text.lower())
            
            if tokens and len(tokens) >= 2:  # Au moins 2 tokens significatifs
                processed.append(' '.join(tokens))
        
        return processed
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Tokenisation basique améliorée sans spaCy"""
        # Supprimer la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Séparer en mots
        words = text.split()
        
        # ✅ STOP WORDS FRANÇAIS COMPLETS
        french_stopwords = {
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
        }
        
        tokens = []
        for word in words:
            # Filtrage strict
            if (len(word) > 3  # Minimum 4 caractères
                and word not in french_stopwords
                and word not in self.custom_stopwords
                and word.isalpha()  # Seulement lettres
                and not word.isdigit()):  # Pas de chiffres
                tokens.append(word)
        
        return tokens
    
    def _remove_stopword_phrases(self, text: str) -> str:
        """Supprime les expressions non informatives"""
        for phrase in self.stopword_phrases:
            text = text.replace(phrase, '')
        return text
    
    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """Extrait les mots-clés avec TF-IDF amélioré"""
        if not texts or len(texts) < 2:
            print("⚠️  Not enough texts for TF-IDF")
            return []
        
        processed_texts = self.preprocess_for_topics(texts)
        
        if len(processed_texts) < 2:
            print("⚠️  Not enough processed texts")
            return []
        
        try:
            # ✅ TF-IDF AMÉLIORÉ
            vectorizer = TfidfVectorizer(
                max_features=150,
                ngram_range=(1, 3),  # Unigrammes, bigrammes ET trigrammes
                min_df=3,  # Doit apparaître dans au moins 3 documents
                max_df=0.7,  # Maximum 70% des documents (pas trop fréquent)
                token_pattern=r'\b[a-zàâäéèêëïîôùûüÿç]{3,}\b',  # Minimum 4 lettres
                strip_accents='unicode'
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculer les scores moyens
            scores = tfidf_matrix.mean(axis=0).A1
            
            # Filtrer les résultats
            filtered_keywords = []
            for i, score in enumerate(scores):
                keyword = feature_names[i]
                
                # ✅ FILTRAGE SUPPLÉMENTAIRE
                # Vérifier que le mot-clé n'est pas dans les stopwords
                if (keyword not in self.custom_stopwords 
                    and not any(stop in keyword for stop in self.custom_stopwords)
                    and score > 0.01):  # Score minimum
                    filtered_keywords.append((keyword, score))
            
            # Trier et prendre les top N
            filtered_keywords.sort(key=lambda x: x[1], reverse=True)
            keywords = filtered_keywords[:top_n]
            
            print(f"✅ Extracted {len(keywords)} meaningful keywords")
            return keywords
        
        except Exception as e:
            print(f"❌ TF-IDF extraction error: {e}")
            return []
    
    def extract_themes_lda(self, texts: List[str], n_topics: int = 5) -> List[Dict]:
        """Extrait les thèmes avec LDA amélioré"""
        processed_texts = self.preprocess_for_topics(texts)
        
        if len(processed_texts) < 10:
            print("⚠️  Not enough texts for LDA, using keywords instead")
            keywords = self.extract_keywords_tfidf(texts, top_n=10)
            return [{
                'theme': kw[0],
                'keywords': [kw[0]],
                'weight': kw[1]
            } for kw in keywords[:5]]
        
        try:
            # ✅ VECTORISATION AMÉLIORÉE
            vectorizer = TfidfVectorizer(
                max_features=150,
                min_df=3,
                max_df=0.7,
                ngram_range=(1, 2),
                token_pattern=r'\b[a-zàâäéèêëïîôùûüÿç]{4,}\b'
            )
            #Matrice Documents-Termes.
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # ✅ LDA AMÉLIORÉ
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=30,  # Plus d'itérations
                learning_method='batch',
                n_jobs=-1  # Utiliser tous les CPU
            )
            
            lda.fit(doc_term_matrix)
            
            # Extraire les thèmes
            themes = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-8:][::-1]  # Top 8 mots
                top_words = [feature_names[i] for i in top_indices]
                
                # ✅ FILTRER LES MOTS NON SIGNIFICATIFS
                meaningful_words = [
                    word for word in top_words 
                    if word not in self.custom_stopwords
                ][:5]
                
                if len(meaningful_words) >= 2:
                    # Créer un nom de thème basé sur les 2-3 mots principaux
                    theme_name = ' + '.join(meaningful_words[:3])
                    
                    themes.append({
                        'theme': theme_name,
                        'keywords': meaningful_words,
                        'weight': float(topic.sum())
                    })
            
            print(f"✅ Extracted {len(themes)} meaningful themes with LDA")
            return themes
        
        except Exception as e:
            print(f"❌ LDA extraction error: {e}")
            return []
    
    def extract_frequent_phrases(self, texts: List[str], min_count: int = 3) -> List[Tuple[str, int]]:
        """Extrait les phrases fréquentes significatives"""
        all_ngrams = []
        
        for text in texts:
            if self.nlp:
                doc = self.nlp(text.lower())
                
                # ✅ EXTRACTION AMÉLIORÉE DE BIGRAMMES
                for i in range(len(doc) - 1):
                    token1, token2 = doc[i], doc[i + 1]
                    
                    # Conditions strictes
                    if (not token1.is_stop and not token2.is_stop
                        and token1.pos_ in ['NOUN', 'ADJ', 'VERB'] 
                        and token2.pos_ in ['NOUN', 'ADJ', 'VERB']
                        and len(token1.text) > 3 and len(token2.text) > 3
                        and token1.lemma_ not in self.custom_stopwords
                        and token2.lemma_ not in self.custom_stopwords):
                        
                        bigram = f"{token1.lemma_} {token2.lemma_}"
                        
                        # Vérifier que ce n'est pas une expression stopword
                        if bigram not in self.stopword_phrases:
                            all_ngrams.append(bigram)
            else:
                # Fallback basique
                words = self._basic_tokenize(text.lower())
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i + 1]}"
                    if bigram not in self.stopword_phrases:
                        all_ngrams.append(bigram)
        
        # Compter et filtrer
        counter = Counter(all_ngrams)
        frequent = [
            (phrase, count) for phrase, count in counter.most_common(30) 
            if count >= min_count and len(phrase.split()) == 2  # Seulement bigrammes complets
        ][:20]
        
        return frequent
    def extract_negative_issues(self, texts: List[str], top_n: int = 10) -> List[Dict]:
        """
        Extrait spécifiquement les problèmes des commentaires négatifs
        avec analyse orientée problèmes
        """
        if not texts or len(texts) < 2:
            print("⚠️  Not enough texts for negative issue extraction")
            return []
        
        # Mots-clés de problèmes à rechercher
        problem_keywords = {
            'logistique': ['salle', 'horaire', 'pause', 'lieu', 'organisation', 'planning'],
            'contenu': ['théorique', 'difficile', 'compliqué', 'dense', 'incompréhensible'],
            'rythme': ['rapide', 'lent', 'court', 'long', 'insuffisant'],
            'pratique': ['manque','manqu' 'peu', 'absence', 'exercice', 'pratique', 'exemple'],
            'technique': ['matériel', 'équipement', 'outil', 'logiciel', 'connexion'],
            'formateur': ['explications', 'pédagogie', 'disponibilité', 'compétence']
        }
        
        # Analyser les problèmes par catégorie
        issues_found = {}
        
        for text in texts:
            text_lower = text.lower()
            
            for category, keywords in problem_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        if category not in issues_found:
                            issues_found[category] = {
                                'count': 0,
                                'keywords': set(),
                                'examples': []
                            }
                        issues_found[category]['count'] += 1
                        issues_found[category]['keywords'].add(keyword)
                        
                        # Garder un exemple de commentaire
                        if len(issues_found[category]['examples']) < 3:
                            issues_found[category]['examples'].append(text[:100])
        
        # Formater le résultat
        result = []
        for category, data in sorted(issues_found.items(), 
                                     key=lambda x: x[1]['count'], 
                                     reverse=True)[:top_n]:
            result.append({
                'issue_category': category,
                'occurrences': data['count'],
                'keywords': list(data['keywords']),
                'severity': 'high' if data['count'] > 5 else 'medium' if data['count'] > 2 else 'low',
                'sample_comments': data['examples']
            })
        
        print(f"✅ Extracted {len(result)} negative issue categories")
        return result


    def extract_negative_keywords(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    
        if not texts or len(texts) < 2:
            print("⚠️ Pas assez de textes pour l'analyse")
            return []
        
        # Traitement préalable des textes
        processed_texts = self.preprocess_for_topics(texts)
        
        if not processed_texts or len(processed_texts) < 2:
            print("⚠️ Pas assez de textes traités")
            return []
        
        try:
            # Initialiser les collections
            keywords = []
            
            # Extraire les mots de chaque texte
            for texte in processed_texts:
                # Utiliser une expression régulière pour trouver les mots
                mots = re.findall(r'\b\w+\b', texte.lower())
                
                # Filtrer les mots non pertinents
                mots_filtres = [
                    mot for mot in mots 
                    if (
                        mot not in self.custom_stopwords and 
                        len(mot) > 2)
                ]
                
                keywords.extend(mots_filtres)
            
            # Compter les fréquences
            compteur = collections.Counter(keywords)
            
            # Calculer les scores de fréquence normalisée
            total_mots = sum(compteur.values())
            mots_avec_scores = []
            
            for mot, compte in compteur.most_common(top_n):
                # Score normalisé (fréquence relative)
                score = round(compte / total_mots, 4) if total_mots > 0 else 0.0
                mots_avec_scores.append((mot, score))
            
            # Trier par score décroissant
            mots_avec_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Limiter au top_n demandé
            resultat = mots_avec_scores[:top_n]
            
            # Journalisation
            if resultat:
                print(f"✅ {len(resultat)} mots-clés négatifs extraits (sur {len(keywords)} mots significatifs)")
                # Afficher les 5 premiers pour débogage
                top_5 = ", ".join([f"{mot}:{score}" for mot, score in resultat[:5]])
                print(f"   Top 5: {top_5}")
            else:
                print("ℹ️ Aucun mot-clé négatif significatif trouvé")
            
            return resultat
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des mots-clés négatifs: {e}")
            import traceback
            traceback.print_exc()
            return []