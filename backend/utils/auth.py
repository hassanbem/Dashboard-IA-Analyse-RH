from ldap3 import Server, Connection

def authenticate_user(matricule: str, password: str) -> bool:
    """Authentification Active Directory Safran"""
    try:
        server = Server('ldap://safran-ad-server.local')
        conn = Connection(
            server,
            user=f'SAFRAN\\{matricule}',
            password=password
        )
        return conn.bind()
    except:
        return False

def check_rh_access(matricule: str) -> bool:
    """Vérifie si utilisateur a accès RH"""
    # Vérifier groupe AD "RH_FORMATION"
    # Retourner True/False
    pass