import smtplib
from email.message import EmailMessage

def send_alert_email(results: dict):
    """Envoyer alerte email si problèmes détectés"""
    
    weak_signals = results.get('weak_signals', [])
    
    if not weak_signals:
        return
    
    # Créer email
    msg = EmailMessage()
    msg['Subject'] = f"⚠️ ALERTE: {len(weak_signals)} signaux faibles détectés"
    msg['From'] = "safran-analysis@noreply.com"
    msg['To'] = "rh-formation@safran.com"
    
    body = f"""
    Analyse quotidienne terminée - ALERTES DÉTECTÉES
    
    Nombre de signaux faibles: {len(weak_signals)}
    
    Détails:
    """
    
    for signal in weak_signals:
        body += f"\n- {signal['issue']} ({signal['occurrences']} occurrences)"
    
    msg.set_content(body)
    
    # Envoyer (configurer SMTP Safran)
    with smtplib.SMTP('smtp.safran.internal') as s:
        s.send_message(msg)