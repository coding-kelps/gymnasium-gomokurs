from gymnasium_gomokurs.domain.gymnasium_gomokurs.service import GomokursEnv
from gymnasium_gomokurs.adapters.manager_interfaces.tcp.manager_interface import *
import numpy as np

logging.getLogger("gymnasium-gomokurs").setLevel(logging.DEBUG)

def main():
    tcp_interface_manager = create_tcp_manager_interface_from_active_connection()

    env = GomokursEnv(tcp_interface_manager)

    while True:
        observation, info = env.reset()

        episode_over = False
        while not episode_over:
            availables = np.flatnonzero(observation["availables"].flat == 1)
            action = np.random.choice(availables)

            observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

            if truncated:
                env.close()

                return

if __name__ == "__main__":
    main()
