import schedule
import time

class DailyBatchImporter:
    """Import quotidien depuis SELIA"""
    
    def __init__(self, source_path: str):
        self.source_path = source_path  # Chemin export SELIA
    
    def import_daily_evaluations(self):
        """Import et traitement J-1"""
        logger.info("Starting daily batch import from SELIA...")
        
        # 1. Récupérer fichier export SELIA
        files = glob.glob(f"{self.source_path}/evaluations_*.csv")
        latest = max(files, key=os.path.getctime)
        
        # 2. Charger et anonymiser
        df = pd.read_csv(latest)
        df_anon = anonymize_advanced(df)
        
        # 3. Analyser automatiquement
        results = analyze_batch(df_anon)
        
        # 4. Générer alerte si signaux faibles
        if results['weak_signals']:
            send_alert_email(results)
        
        logger.info("Daily import completed")
    
    def schedule_daily(self, time_str="06:00"):
        """Planifier import quotidien"""
        schedule.every().day.at(time_str).do(
            self.import_daily_evaluations
        )