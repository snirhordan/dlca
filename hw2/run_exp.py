# srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 64 -L 2 -P 2 -H 100
import argparse
import subprocess
from typing import List


EXPERIMENT_LIST = ['1.1', '1.2', '1.3', '1.4', '2', 'all']


def generate_test_name(test_num) -> str:
    return f'exp{test_num.replace(".", "_")}'


def generate_test_cli(args) -> List[str]:
    commands = []
    for experiment in args.experiment:
        if experiment not in EXPERIMENT_LIST:
            raise ValueError()

        if experiment.lower() in ['1.1', 'all']:
            k_values = [32, 64]
            l_values = [2, 4, 8, 16]
            for K in k_values:
                for L in l_values:
                    default_command = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n'
                    test_name = generate_test_name(experiment)
                    k_str = '-K ' + str(K)
                    l_str = f'-L {L}'
                    p_str = '-P 8' #2
                    h_str = '-H 100'
                    commands.append(f'{default_command} {test_name} {k_str} {l_str} {p_str} {h_str}')

        if experiment.lower() in ['1.2', 'all']:
            k_values = [32, 64, 128, 256]
            l_values = [2, 4, 8]
            for L in l_values:
                for K in k_values:
                    default_command = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n'
                    test_name = generate_test_name(experiment)
                    k_str = '-K ' + str(K)
                    l_str = f'-L {L}'
                    p_str = '-P 8'#2
                    h_str = '-H 100'
                    commands.append(f'{default_command} {test_name} {k_str} {l_str} {p_str} {h_str}')

        if experiment.lower() in ['1.3', 'all']:
            k_values = [64, 128, 256]
            l_values = [1, 2, 3, 4]
            for L in l_values:
                default_command = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n'
                test_name = generate_test_name(experiment)
                k_str = '-K ' + ' '.join([str(k) for k in k_values])
                l_str = f'-L {L}'
                p_str = '-P 8' #2
                h_str = '-H 100'
                commands.append(f'{default_command} {test_name} {k_str} {l_str} {p_str} {h_str}')

        if experiment.lower() in ['1.4', 'all']:
            k_values = [[32], [64, 128, 256]]
            l_values = [[8, 16, 32], [2, 4, 8]]
            for i in range(2):
                for L in l_values[i]:
                    default_command = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n'
                    test_name = generate_test_name(experiment)
                    k_str = '-K ' + ' '.join([str(k) for k in k_values[i]])
                    l_str = f'-L {L}'
                    p_str = '-P 8' #2
                    h_str = '-H 100'
                    m_str = '-M resnet'
                    commands.append(f'{default_command} {test_name} {k_str} {l_str} {p_str} {h_str} {m_str}')

        if experiment.lower() in ['2', 'all']:
            k_values = [32, 64, 128]
            l_values = [3, 6, 9, 12]
            for L in l_values:
                default_command = 'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n'
                test_name = generate_test_name(experiment)
                k_str = '-K ' + ' '.join([str(k) for k in k_values])
                l_str = f'-L {L}'
                p_str = '-P 8'
                h_str = '-H 100'
                m_str = '-M ycn'
                commands.append(f'{default_command} {test_name} {k_str} {l_str} {p_str} {h_str} {m_str}')

    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-exp', '--experiment',
        required=True,
        type=str,
        nargs='+',
        choices=EXPERIMENT_LIST,
        help='The experiment/s to run.'
    )
    args = parser.parse_args()
    commands = generate_test_cli(args)
    print(args, '\n'.join(commands))
    for i, command in enumerate(commands):
        print(f'---- Running experiment {i+1}/{len(commands)} ----')
        subprocess.run(command.split())
    print(f'---- {len(commands)} experiments completed ----')


if __name__ == '__main__':
    main()
