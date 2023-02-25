################################################################
# Machine Learning
# Assignment 1 Starting Code
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################


import argparse

import numpy
import numpy as np
# import statistics as stats
import matplotlib.pyplot as plt

################################################################
# Metrics and visualization
################################################################


def conf_matrix(class_matrix, data_matrix, title=None, print_results=False):

    # Initialize confusion matrix with 0's
    confusions = np.zeros((2, 2), dtype=int)

    # Generate output/target pairs, convert to list of integer pairs
    # * '-1' indexing indicates the final column
    # * list(zip( ... )) combines the output and target class label columns into a list of pairs
    out_target_pairs = [
        (int(out), int(target))
        for (out, target) in list(zip(class_matrix[:, -1], data_matrix[:, -1]))
    ]

    # Use output/target pairs to compile confusion matrix counts
    for (out, target) in out_target_pairs:
        confusions[out][target] += 1

    # Compute recognition rate
    inputs_correct = confusions[0][0] + confusions[1][1]
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total * 100

    if print_results:
        if title:
            print("\n>>>  " + title)
        print(
            "\n    Recognition rate (correct / inputs):\n    ", recognition_rate, "%\n"
        )
        print("    Confusion Matrix:")
        print("              0: Blue-True  1: Org-True")
        print("---------------------------------------")
        print("0: Blue-Pred |{0:12d} {1:12d}".format(confusions[0][0], confusions[0][1]))
        print("1: Org-Pred  |{0:12d} {1:12d}".format(confusions[1][0], confusions[1][1]))

    return (recognition_rate, confusions)


def draw_results(data_matrix, class_fn, title, file_name):

    # Fix axes ranges so that X and Y directions are identical (avoids 'stretching' in one direction or the other)
    # Use numpy amin function on first two columns of the training data matrix to identify range
    pad = 0.25
    min_tick = np.amin(data_matrix[:, 0:2]) - pad
    max_tick = np.amax(data_matrix[:, 0:2]) + pad
    plt.xlim(min_tick, max_tick)
    plt.ylim(min_tick, max_tick)

    ##################################
    # Grid dots to show class regions
    ##################################

    axis_tick_count = 75
    x = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    y = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    (xx, yy) = np.meshgrid(x, y)
    grid_points = np.concatenate(
        (xx.reshape(xx.size, 1), yy.reshape(yy.size, 1)), axis=1
    )

    class_out = class_fn(grid_points)

    # Separate rows for blue (0) and orange (1) outputs, plot separately with color
    blue_points = grid_points[np.where(class_out[:, 1] < 1.0)]
    orange_points = grid_points[np.where(class_out[:, 1] > 0.0)]

    plt.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        marker=".",
        s=1,
        facecolors="blue",
        edgecolors="blue",
        alpha=0.4,
    )
    plt.scatter(
        orange_points[:, 0],
        orange_points[:, 1],
        marker=".",
        s=1,
        facecolors="orange",
        edgecolors="orange",
        alpha=0.4,
    )

    ##################################
    # Decision boundary (black line)
    ##################################

    #  TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # MISSING -- add code to draw class boundaries
    # ONE method for ALL classifier types

    if class_fn == linear_classifier: # todo - how do we do function comparisons?
        pass
        bound = 0.5
        # we would need to reobtain the optimal beta vector, just copy the code as-is
        # from there, not sure
        # we need to find a closed-form solution to 0.5 = X @ B, but I don't think this question is even coherent

    """
        FOR LIN MODELS
        you need to solve 
            0.5 = X^ @ B
                where X is UNKNOWN
                B is the optimal beta derived from the input matrix X (not the X^ here)
            
            score (Y) = X^ @ B
                is always 0.5 for lin model
                
        FOR KNN
            above formula doesn't apply
            
            one method: if you run less than 10 seconds, no deductions
                the resolution of the grid is NxM
                for each point in the grid:
                    predict on the point and get score for the point
                    set a margin (make coeff of K?) where the point will be judged to be on the boundary
                    plot the point as black
                    
                    
                    ^^^ breaks down at k = 1
                        observe the shift in class predictions compared to the previous point
                        scan over the rows 
                
            
    """
    pass


    ##################################
    # Show training samples
    ##################################

    # Separate rows for blue (0) and orange (1) target inputs, plot separately with color
    blue_targets = data_matrix[np.where(data_matrix[:, 2] < 1.0)]
    orange_targets = data_matrix[np.where(data_matrix[:, 2] > 0.0)]

    plt.scatter(
        blue_targets[:, 0],
        blue_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
    )
    plt.scatter(
        orange_targets[:, 0],
        orange_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="darkorange",
    )
    ##################################
    # Add title and write file
    ##################################

    # Set title and save plot if file name is passed (extension determines type)
    plt.title(title)
    plt.savefig(file_name)
    print("\nWrote image file: " + file_name)
    plt.close()


################################################################
# Interactive Testing
################################################################
def test_points(data_matrix, beta_hat):

    print("\n>> Interactive testing for (x_1,x_2) inputs")
    stop = False
    while True:
        x = input("\nEnter x_1 ('stop' to end): ")

        if x == "stop":
            break
        else:
            x = float(x)

        y = float(input("Enter x_2: "))
        k = int(input("Enter k: "))

        lc = linear_classifier(beta_hat)
        knn = knn_classifier(k, data_matrix)

        print("   least squares: " + str(lc(np.array([x, y]).reshape(1, 2))))
        print("             knn: " + str(knn(np.array([x, y]).reshape(1, 2))))


################################################################
# Classifiers
################################################################

def optimal_beta_finder(input_matrix):
    # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
    # -- as defined in Hastie equation (2.2)
    row_count = input_matrix.shape[0]
    col_count = input_matrix.shape[1]

    # ---- Find the optimal beta vector
    x = numpy.matrix.copy(input_matrix)  # Don't harm the data

    # define y
    y = x[:, -1]  # Extract y (output) column from input matrix

    # expunge y from input matrix
    x = np.delete(x, col_count - 1, 1)  # args: delete from x, at the [col_count - 1] index, a column.

    # Dose x with a '1' in the leftmost column, needed to normalize with bias
    filler_array = np.ones((row_count, 1))
    x = np.column_stack((filler_array, x))

    # define transpose of x, 1 biasing included
    x_t = x.transpose()  # X^T

    left_product = x_t @ x  # (X^T @ X)

    left_inverse = np.linalg.inv(left_product)  # (X^T @ X)^-1

    right_product = x_t @ y  # (X^T @ y)

    beta = left_inverse @ right_product  # B = (X^T @ X)^-1 @ (X^T @ y)

    return beta


def linear_classifier(weight_vector): # weight vector = beta vector
    # Constructs a linear classifier

    def classifier(input_matrix): # todo type numpy array?
        # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
        # -- as defined in Hastie equation (2.2)
        row_count = input_matrix.shape[0]
        col_count = input_matrix.shape[1]

        # print("start:",input_matrix.shape)

        # NOTE: we have access to weight_vector, can call it directly


        # ---- Find the optimal beta vector
        x = numpy.matrix.copy(input_matrix) # Don't harm the original data.

        # expunge y from input matrix
        if x.shape[1] == 3:
            x = np.delete(x, col_count - 1, 1) # args: delete from x, at the [col_count - 1] index, a column.
            # print("deleted col:", x.shape)

        # # Dose x with a '1' in the leftmost column, needed to normalize with bias
        filler_array = np.ones((row_count,1))
        x = np.column_stack((filler_array, x))
        # print("filled col:",x.shape)


        # ----- Find the result Y^

        # print('shape of x:', x.shape)
        # print('shape of beta:', weight_vector.shape)

        y_hat = x @ weight_vector # Y^ = X^T @ B, should be good as is

        # Add the result as another column: If >=0.5, 1, else 0
        y_hat = np.column_stack(
            (y_hat, np.zeros(
                    (y_hat.shape[0],1)
                )
            )
        )
        for score_index in range(y_hat.shape[0]):
            y_hat[score_index][1] = 1 if y_hat[score_index][0] >= 0.5 else 0

        return y_hat

    return classifier


def knn_classifier(k, data_matrix): # Data matrix is the training dat
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix):


        # Setup names and aliases
        training_data = data_matrix # rename for clarity, data_matrix is the training data
        prediction_matrix = input_matrix # rename for clarity
        prediction_row_count = prediction_matrix.shape[0]
        training_row_count = training_data.shape[0]

        # Initialize data structures
        distance_matrix = numpy.zeros((prediction_row_count,prediction_row_count)) # dist matrix,
        k_matrix = numpy.zeros((prediction_row_count,k)) #holds the k selections per predict case

        # [calculated score, classification]. Returned item.
        k_classification = numpy.zeros((prediction_row_count, 2))

        map_distance_classification = {} # Map calculated distance to the classification

        # Begin nearest neighbored classification - one of these loops = one prediction processed
        for prediction_index in range(0, prediction_row_count): # for each row, calc dist to each point in training_data

            # Extract features to predict upon
            pred_x = prediction_matrix[prediction_index][0]
            pred_y = prediction_matrix[prediction_index][1]



            # Iterate over all training data.
            for training_index in range(training_row_count):
                # Extract features and output from training data sample
                training_x = training_data[training_index][0]
                training_y = training_data[training_index][1]
                training_item_classification = training_data[training_index][2]

                # Calculate Euclidean Distance
                distance = (pred_x -training_x)**2 + (pred_y - training_y)**2
                distance_matrix[prediction_index][training_index] = distance # Store distance

                # Map the calculated distance to the classification
                map_distance_classification[distance] = training_item_classification # TODO on loop 199, something is bad with the key


            # Sort the calculated distance to be able to slice out k items
            distance_matrix[prediction_index].sort()

            # Slice out k items
            k_selections = distance_matrix[prediction_index][0:k]

            # Need to undo mapping, correlate the distances back to their training sample
            selected_k_classifications = []
            for c in k_selections:
                try:
                    selected_k_classifications.append(map_distance_classification[c])
                except KeyError as e:
                    print(e)


            # Sum the classifications - this is just adding 1s and 0s
            k_score = sum(selected_k_classifications)

            # If the mean class in the k set is 0.5 or greater, predict this item to be true, else false
            classification = 1 if k_score >= 0.5 else 0

            # Copy the k selections into a greater matrix for storage
            k_matrix[prediction_index] =  k_selections

            # Fill the output matrix - 0th column is the mean classification of the k selections, 1st is the prediction
            k_classification[prediction_index][0] = k_score
            k_classification[prediction_index][1] = classification

        return k_classification

        """
        distance_matrix = numpy.zeros((row_count,row_count)) # for N rows in the input matrix, create NxN dist matrix

        for row_index in range(row_count):
            row_element = x_matrix[row_index] 
            x = row_element[0]
            y = row_element[1]

            for comparison_index in range(row_count):
                comparison_element = x_matrix[row_element]
                a = comparison_element[0]
                b = comparison_element[1]

                distance = (x - a)^2 + (y - b)^2
                distance_matrix[row_index][comparison_index] = distance


        prediction_matrix = input_matrix
        for pred_item in prediction_matrix: # EXTRACT K ITEMS FROM THE DISTANCE MATRIX
            for i in range(0,row_count):

                row = distance_matrix[i] # extract a row to work on
                row.sort()
                k_array = row[0:k]

                class_sum = 0 #
                for item in k_array: # REVERSE MAP BACK TO THE TRAINING DATA FROM THESE POINTS
                    item_index = row.index(item)
        """

        """   -------
                knn is just a geometric rep of probability, vals associated with 'good' spaces are assumed good
                
                k defs the size of that 'good' space - the 'neighborhood'
                    k = num of neighbors
                        always at least 1, 1 is the self
                        
                the algo is that we calc euclidean dist from the input to all other points in the training data
                
                ^^^^ there is no given algo for this
                    you need to figure it out yourself
                    
                least squares does not relate eto this at all 
                
                you can do a distance matrix
                    using euc dist formula
                    
                indexing is important
                
                to compare point 0 to point 4,
                    go to row 4
                        go to col 4
                            calc or fetch the value
                                calc w (x - a)^2 + (y - b)^2
                                
                get the k closest vals
                fetch their values from the matrix (last col of row)
                do ratio of True to False (training is binary scored)
                
                output is Wx2 matrix
                    W is the number of input samples we are predicting on
                    first col is the calc'd score
                        second is the classified class (0 or 1)
                    
            
        """


    return classifier


################################################################
# Main function
################################################################


def main():
    # Process arguments using 'argparse'
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="numpy data matrix file containing samples")
    args = parser.parse_args()

    # Load data
    data_matrix = np.load(args.data_file)
    (confm, rr) = conf_matrix(
        data_matrix, data_matrix, "Data vs. Itself Sanity Check", print_results=True
    )

    # Construct linear classifier
    # finds optimal beta --- NOTE: USE YOUR OWN FUNCTION HERE !!!!!!!!!!!
    optimal_beta = optimal_beta_finder(data_matrix)
    # lc = linear_classifier(np.array([1,1,1]))  # original code, replace the array with the optimal beta
    lc = linear_classifier(optimal_beta)
    lsc_out = lc(data_matrix)

    # Compute results on training set
    conf_matrix(lsc_out, data_matrix, "Least Squares", print_results=True)
    draw_results(data_matrix, lc, "Least Squares Linear Classifier", "ls.pdf")

    # Nearest neighbor
    for k in [1, 15]:
        knn = knn_classifier(k, data_matrix)
        knn_out = knn(data_matrix)
        conf_matrix(knn_out, data_matrix, "knn: k=" + str(k), print_results=True)
        draw_results(
            data_matrix,
            knn,
            "k-NN Classifier (k=" + str(k) + ")",
            "knn-" + str(k) + ".pdf",
        )

    # Interactive testing
    test_points(data_matrix, np.array([1,1,1]))


if __name__ == "__main__":
    main()
