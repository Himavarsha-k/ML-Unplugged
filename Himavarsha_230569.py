{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment No. 2 Solutions\n",
    "## Machine Learning - Decision Trees"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem 1: Gini Index Calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gini_index(positive, negative):\n",
    "    total = positive + negative\n",
    "    if total == 0:\n",
    "        return 0\n",
    "    p_positive = positive / total\n",
    "    p_negative = negative / total\n",
    "    return 1 - (p_positive**2 + p_negative**2)\n",
    "\n",
    "# Original dataset\n",
    "total_positive = 220\n",
    "total_negative = 80\n",
    "total_samples = total_positive + total_negative\n",
    "\n",
    "gini_before = gini_index(total_positive, total_negative)\n",
    "print(f\"Gini index before splitting: {gini_before:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# After split results\n",
    "left_positive = 90\n",
    "left_negative = 10\n",
    "left_total = left_positive + left_negative\n",
    "\n",
    "right_positive = 100\n",
    "right_negative = 100\n",
    "right_total = right_positive + right_negative\n",
    "\n",
    "# Calculate Gini for subsets\n",
    "gini_left = gini_index(left_positive, left_negative)\n",
    "gini_right = gini_index(right_positive, right_negative)\n",
    "\n",
    "# Weighted Gini\n",
    "weighted_gini = (left_total/total_samples)*gini_left + (right_total/total_samples)*gini_right\n",
    "\n",
    "print(f\"Left subset Gini: {gini_left:.4f}\")\n",
    "print(f\"Right subset Gini: {gini_right:.4f}\")\n",
    "print(f\"Weighted Gini after split: {weighted_gini:.4f}\")\n",
    "\n",
    "if weighted_gini < gini_before:\n",
    "    print(\"Split improves purity\")\n",
    "else:\n",
    "    print(\"Split does not improve purity\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem 2: Regression Tree with SSE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Dataset\n",
    "data = {\n",
    "    'X1': [1, 2, 3, 4, 5, 6, 7, 8],\n",
    "    'X2': [5, 6, 8, 10, 12, 15, 18, 20],\n",
    "    'Y': [10, 12, 15, 18, 21, 25, 28, 30]\n",
    "}\n",
    "\n",
    "def calculate_sse(data, feature, split_val):\n",
    "    left = [y for x, y in zip(data[feature], data['Y']) if x <= split_val]\n",
    "    right = [y for x, y in zip(data[feature], data['Y']) if x > split_val]\n",
    "    \n",
    "    sse_left = sum((y - np.mean(left))**2 for y in left) if left else 0\n",
    "    sse_right = sum((y - np.mean(right))**2 for y in right) if right else 0\n",
    "    \n",
    "    return sse_left + sse_right"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find best split for X1\n",
    "best_split_x1 = None\n",
    "best_sse_x1 = float('inf')\n",
    "\n",
    "for i in range(len(data['X1'])-1):\n",
    "    split_val = (data['X1'][i] + data['X1'][i+1]) / 2\n",
    "    current_sse = calculate_sse(data, 'X1', split_val)\n",
    "    \n",
    "    if current_sse < best_sse_x1:\n",
    "        best_sse_x1 = current_sse\n",
    "        best_split_x1 = split_val\n",
    "\n",
    "print(f\"Best X1 split: {best_split_x1} with SSE: {best_sse_x1:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find best split for X2\n",
    "best_split_x2 = None\n",
    "best_sse_x2 = float('inf')\n",
    "\n",
    "for i in range(len(data['X2'])-1):\n",
    "    split_val = (data['X2'][i] + data['X2'][i+1]) / 2\n",
    "    current_sse = calculate_sse(data, 'X2', split_val)\n",
    "    \n",
    "    if current_sse < best_sse_x2:\n",
    "        best_sse_x2 = current_sse\n",
    "        best_split_x2 = split_val\n",
    "\n",
    "print(f\"Best X2 split: {best_split_x2} with SSE: {best_sse_x2:.2f}\")\n",
    "\n",
    "# Determine best overall split\n",
    "if best_sse_x1 < best_sse_x2:\n",
    "    print(f\"First split should be on X1 at {best_split_x1}\")\n",
    "else:\n",
    "    print(f\"First split should be on X2 at {best_split_x2}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualization\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.scatter(data['X1'], data['Y'], label='X1 vs Y', color='blue')\n",
    "plt.scatter(data['X2'], data['Y'], label='X2 vs Y', color='red')\n",
    "\n",
    "if best_sse_x1 < best_sse_x2:\n",
    "    plt.axvline(x=best_split_x1, color='blue', linestyle='--', \n",
    "                label=f'Best split on X1 at {best_split_x1}')\n",
    "else:\n",
    "    plt.axvline(x=best_split_x2, color='red', linestyle='--', \n",
    "                label=f'Best split on X2 at {best_split_x2}')\n",
    "\n",
    "plt.xlabel('X1 (blue) and X2 (red)')\n",
    "plt.ylabel('Y')\n",
    "plt.title('Regression Tree - Best First Split')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 }
}