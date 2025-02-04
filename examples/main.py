from gymnasium_gomokurs.domain.gymnasium_gomokurs.service import GomokursEnv
from gymnasium_gomokurs.adapters.manager_interfaces.tcp.manager_interface import *

def main():
    tcp_interface_manager = create_tcp_manager_interface_from_active_connection()

    env = GomokursEnv(tcp_interface_manager)

    while True:
        pass

if __name__ == '__main__':
    main()
