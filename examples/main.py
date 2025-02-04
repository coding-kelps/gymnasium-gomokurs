from gymnasium_gomokurs.domain.gymnasium_gomokurs.service import GomokursEnv
from gymnasium_gomokurs.adapters.manager_interfaces.tcp.manager_interface import *

def main():
    tcp_interface_manager = create_tcp_manager_interface_from_active_connection()

    env = GomokursEnv(tcp_interface_manager)

    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()        
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()

if __name__ == '__main__':
    main()
