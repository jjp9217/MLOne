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

def linear_classifier(weight_vector):
    # Constructs a linear classifier
    def classifier(input_matrix): # todo type numpy array?
        # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
        # -- as defined in Hastie equation (2.2)
        row_count = input_matrix.shape[0]
        col_count = input_matrix.shape[1]

        # NOTE: we have access to weight_vector, can call it directly


        # ---- Find the optimal beta vector
        x = input_matrix # rename to match formula for clarity

        # define y
        y = x[:,-1] # Extract y (output) column from input matrix
        # expunge y from input matrix
        x = np.delete(x, col_count - 1, 1) # args: delete from x, at the [col_count - 1] index, a column.

        # Dose x with a '1' in the leftmost column, needed to normalize with bias
        filler_array = np.ones((row_count,1))
        x = np.column_stack((filler_array, x))

        # define transpose of x, 1 biasing included
        x_t = x.transpose() # X^T

        left_product = x_t @ x  # (X^T @ X)

        left_inverse = np.linalg.inv(left_product) #(X^T @ X)^-1

        right_product = x_t @ y # (X^T @ y)

        beta_vector = left_inverse @ right_product #B = (X^T @ X)^-1 @ (X^T @ y)

        # ----- Find the result Y^

        y_hat = x @ beta_vector # Y^ = X^T @ B, should be good as is

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


def knn_classifier(k, data_matrix):
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix):

        # inputs are two var pairs. one feature per row 
        (input_count, _) = input_matrix.shape # todo delete?

        # TODO !!!!!!!!!!!!!!!!!!!! exactly what are we training on - the input_matrix, or the data_matrix?
        x_matrix = data_matrix  # rename
        row_count = x_matrix.shape[0]



        # do the euclidean distance matrix
        # first, copy the final y (output) column from each row
        output_column = input_matrix[:, -1]

        # create the distance matrix
        distance_matrix = numpy.zeros((row_count,row_count)) # for N rows in the input matrix, create NxN dist matrix

        for row_index in range(row_count): # FORM THE DISTANCE MATRIX FROM X MATRIX
            row_element = x_matrix[row_index]
            x = row_element[0]
            y = row_element[1]

            for comparison_index in range(row_count):
                comparison_element = x_matrix[row_element]
                a = comparison_element[0]
                b = comparison_element[1]

                distance = (x - a)^2 + (y - b)^2
                distance_matrix[row_index][comparison_index] = distance


        for i in range(0,row_count): # EXTRACT K ITEMS FROM THE DISTANCE MATRIX
            row = distance_matrix[i] # extract a row to work on
            k_array = []
            for item in row:
                if len(k_array) == k: # can we fit this item in?
                    k_max = max(k_array)
                    if item < k_max: # if this value is smaller than the largest element, we can fit it in
                        k_array.remove(max(k_array)) # take off the largest item
                        k_array.append(item)

                else:
                    k_array.append(item)

            # now get the index of each item to link back to the input data

            class_array = []
            for item in k_array: # CLASS COMPARE THE K ARRAY
                item_index = row.index(item)
                # TODO reevaluate once we have the data/input confusion clarified

                # TODO - I don't have sufficient information to determine what to do.
                """
                    We going to, presumably have pairs of coordinates to predict upon. 
                    I have no way as of 10PM 2-22-23 to disambiguate which matrix actually contains this data.
                """

                # now we have the (i,j) coordinate. we know where in the orinal matrix this value is.



        # now do K selections
        # for each row
        # get the K smallest selections
        # reverse index the elements back to the source data
        # extract the y (class) from the kth elements
        # do majority vote from the classes to predict
        """
            ex: 4 0s, 6 1s -> predict this to be 1
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

        # TODO ESTABLISH DISTANCE MATRIX


        # REVISE: Always choose class 1
        scores_classes = np.ones((input_count,2))

        classified_matrix = np.array([[0,0],[0,0]]) # TODO populate this with data
        

        # Return N x 2 result array
        return scores_classes

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
    lc = linear_classifier(np.array([1,1,1]))
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
