import random

def get_random_proxy(username_file: str =  "../proxy_user.txt", 
                     pw_file: str = "../proxy_pw.txt", 
                     servers_path: str = "../servers.txt",
                     blocked_servers_path: str = "../blocked_servers.txt",
                     port: int = 89) -> str:
    # get username
    with open(username_file, "r") as f:
        user = f.read().strip()

    # get password
    with open(pw_file, "r") as f:
        pw = f.read().strip()
    
    def clean_server_list(servers: list) -> list:
        servers = [s.strip().replace("\n", "") for s in servers]
        servers = [s for s in servers if s.endswith(".com")]
        return servers
    
    # get server list
    with open(servers_path, "r") as f:
        servers = clean_server_list(f.readlines())
    
    # get blocked servers
    with open(blocked_servers_path, "r") as f:
        blocked_servers = clean_server_list(f.readlines())
    
    # get random server
    servers = [s for s in servers if s not in blocked_servers]
    server = random.choice(servers)
    
    print(f"Using proxy: {server}:{port}")
    return f"https://{user}:{pw}@{server}:{port}"

def log_blocked_servers(server: str, blocked_servers_path: str = "../blocked_servers.txt") -> None:
    with open(blocked_servers_path, "a") as file:
        file.write(server + "\n")
