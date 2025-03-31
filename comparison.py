import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

adults = pd.read_csv('./data/adult100.csv')
generated_adults = pd.read_csv('./data/generated_adult100_dp.csv')

def plot_continuous_variable(column_name, bins=20):
    plt.figure(figsize=(10, 6))

    plt.hist(adults[column_name], bins=bins, alpha=0.5, label='Original Adults',
             color='blue', edgecolor='black', density=True)
    plt.hist(generated_adults[column_name], bins=bins, alpha=0.5, label='Generated Adults',
             color='green', edgecolor='black', density=True)

    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.title(f'{column_name} Distribution Comparison')
    plt.legend()
    plt.show()


plot_continuous_variable('age')

def plot_discrete_variables(column_name, labels):
    variable_counts_adults = adults[column_name].value_counts(normalize=True)
    variable_counts_generated = generated_adults[column_name].value_counts(normalize=True)


    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, [variable_counts_adults.get(label, 0) for label in labels], width, label='Original Adults',
           color='blue', alpha=0.5)
    ax.bar(x + width / 2, [variable_counts_generated.get(label, 0) for label in labels], width, label='Generated Adults',
           color='green', alpha=0.5)

    ax.set_xlabel(column_name)
    ax.set_ylabel('Proportion')
    ax.set_title(column_name + 'Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

plot_discrete_variables('race', ['Black', 'White', 'Amer-Indian-Eskimo', 'Other', 'Asian-Pac-Islander'])
plot_discrete_variables('sex', ['Male', 'Female'])
plot_discrete_variables('workclass', ['Private', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
plot_discrete_variables('income', ['<=50K', '>50K'])


def plot_joint_distribution(col1, col2):
    joint_original = adults.groupby([col1, col2]).size().unstack().fillna(0)
    joint_generated = generated_adults.groupby([col1, col2]).size().unstack().fillna(0)

    joint_original = joint_original / joint_original.sum().sum()
    joint_generated = joint_generated / joint_generated.sum().sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(joint_original, cmap='Blues', aspect='auto')
    axes[0].set_xticks(np.arange(len(joint_original.columns)))
    axes[0].set_xticklabels(joint_original.columns, rotation=45)
    axes[0].set_yticks(np.arange(len(joint_original.index)))
    axes[0].set_yticklabels(joint_original.index)
    axes[0].set_title(f'Original {col1} vs {col2}')

    axes[1].imshow(joint_generated, cmap='Greens', aspect='auto')
    axes[1].set_xticks(np.arange(len(joint_generated.columns)))
    axes[1].set_xticklabels(joint_generated.columns, rotation=45)
    axes[1].set_yticks(np.arange(len(joint_generated.index)))
    axes[1].set_yticklabels(joint_generated.index)
    axes[1].set_title(f'Generated {col1} vs {col2}')

    plt.show()

plot_joint_distribution('sex', 'income')
plot_joint_distribution('workclass', 'income')
plot_joint_distribution('race', 'income')


plt.show()
