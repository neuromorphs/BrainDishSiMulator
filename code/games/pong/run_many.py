import subprocess

players = ["Simon", "Hugo", "LIFELSE", "Hebbian_Simon"]

for player in players:
    command = [
        "python",
        "PongUI.py",
        "--player",
        player,
        "--num_repeat",
        "5",
        "--num_episodes",
        "70",
        "--verbose",
        "2",
        "--fps",
        "24",
        "--simulation_only",
    ]

    subprocess.run(command)
