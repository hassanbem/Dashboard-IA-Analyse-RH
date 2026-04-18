import pandas as pd
import hashlib
import re
from typing import List
from loguru import logger

def hash_id(id_value: str) -> str:
    """Hash un ID de manière irreversible"""
    if pd.isna(id_value):
        return "UNKNOWN"
    return hashlib.sha256(str(id_value).encode()).hexdigest()[:16]

def remove_pii(text: str) -> str:
    """Supprime les informations personnelles (PII) d'un texte"""
    if not isinstance(text, str) or not text:
        return text
    # Supprimer les emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Supprimer les numéros de téléphone (formats FR)
    text = re.sub(r'\b0[1-9](?:\s?\d{2}){4}\b', '[PHONE]', text)
    text = re.sub(r'\b\+33\s?[1-9](?:\s?\d{2}){4}\b', '[PHONE]', text)
    # Supprimer les numéros d'employé
    text = re.sub(r'\b[A-Z]{2}\d{4,}\b', '[EMPLOYEE_ID]', text)
    return text

def anonymize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fonction principale d'anonymisation"""
    logger.info("Starting data anonymization (RGPD compliance)")
    df = df.copy()
    
    if 'evaluation_id' in df.columns:
        df['evaluation_id'] = df['evaluation_id'].apply(hash_id)
    
    if 'formateur_id' in df.columns:
        df['formateur_id'] = df['formateur_id'].apply(hash_id)
    
    if 'commentaire' in df.columns:
        df['commentaire'] = df['commentaire'].apply(remove_pii)

    logger.info("Data anonymization completed")
    return df

def anonymize_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Anonymisation Phase 2 conforme Safran"""
    
    # Supprimer colonnes sensibles si présentes
    sensitive_cols = ['nom', 'prenom', 'email', 'telephone', 
                     'numero_employe', 'adresse']
    df = df.drop(columns=[c for c in sensitive_cols if c in df.columns])
    
    # Pseudonymiser IDs
    df['evaluation_id'] = df['evaluation_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )
    df['formateur_id'] = df['formateur_id'].apply(
        lambda x: f"FORM_{hashlib.sha256(str(x).encode()).hexdigest()[:8]}"
    )
    
    # Généraliser dates (garder mois/année uniquement)
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Supprimer PII des commentaires
    df['commentaire'] = df['commentaire'].apply(remove_all_pii)
    
    return df