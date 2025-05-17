import random
import requests
from swiftshadow import QuickProxy
import yt_dlp

def get_random_proxy(mode: str = "credentials") -> str:
    """
    Get a random proxy from the QuickProxy service.
    """
    if mode == "credentials":
        return get_random_proxy_with_credentials()
    elif mode == "file":
        return get_random_proxy_from_file()
    else:
        ip_port, protocol = QuickProxy()  # Output: http://<ip>:<port>
        return f"{protocol}://{ip_port}"

def get_random_proxy_from_file(file_path: str = "../proxies.txt") -> str:
    """
    Get a random proxy from a file.
    """
    with open(file_path, "r") as f:
        proxies = f.readlines()
    proxy = random.choice(proxies).strip()
    return f"http://{proxy}"

def get_random_proxy_with_credentials(username_file: str =  "../proxy_user.txt", 
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

def test_proxy_connection(proxy_url: str, test_url: str = "https://httpbin.org/ip", timeout: int = 5) -> bool:
    proxy = {
        "http": proxy_url,
        "https": proxy_url,
    }
    try:
        response = requests.get(test_url, proxies=proxy, timeout=timeout)
        if response:
            test_video_url = "https://www.youtube.com/watch?v=E5XxizdMKRk"  # yt-dlp test video

            ydl_opts = {
                'proxy': proxy,
                'format': 'worst',  # Smallest file for testing
                'outtmpl': 'test_video.mp4',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }

            print(f"Testing download using proxy: {proxy}")

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([test_video_url])
                print("✅ Download succeeded. Proxy is working.")
                return True
            except Exception as e:
                print(f"❌ Download failed. Proxy may not be working.\nError: {e}")

                print("Proxy test succeeded:", response.json())
                return False
    except requests.RequestException as e:
        print(f"Proxy test failed: {e}")
        return False

def test_random_proxy_connection() -> None:
    proxy = get_random_proxy()
    server = proxy.split("@")[-1].split(":")[0]  # extract server address from proxy URL
    if test_proxy_connection(proxy):
        print(f"✅ Proxy {server} is working.")
    else:
        print(f"❌ Proxy {server} failed. Logging as blocked.")
        log_blocked_servers(server)
        
if __name__ == "__main__":
    test_random_proxy_connection()