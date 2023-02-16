################################################################
# Machine Learning
# Assignment 1 Starting Code
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################


import argparse
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

        # NOTE: we have access to weight_vector, can call it directly


        # ---- Find the optimal beta vecotr
        x = input_matrix # rename to match formula for clarity

        # TODO YOU NEED TO DOSE X WITH A 1 IN LEFT COL, PUSH ALL RESULT RIGHT
        x_t = x.transpose() # X^T

        left_product = x_t @ x  # (X^T @ X)

        left_inverse = np.linalg.inv(left_product) #(X^T @ X)^-1

        # define y, is contained

        # TODO DEF y

        y = None # TODO FIX

        right_product = x_t @ y

        beta_vector = left_inverse @ right_product #B = (X^T @ X)^-1 @ (X^T @ y)

        # ----- Find the result Y^

        y_hat = x_t @ beta_vector # Y^ = X^T @ B, should be good as is


        """
                ok
                B = (X^T @ X)^-1 @ (X^T @ y)
                Y^ = X^T @ B
                
                outside of accomplishing the above (and dosing X with 1) there is no more logic needed
          
                    where y is the vector of all outputs
                    
                    lin models work by weighting the features (inputs) by the weights (beta vec)
                    
                    the hats on vars are just for differention, there's no specific concept implied           
                    
                    in the data setup, we always have N X 3 - 2 features map to one output
                        last col is always output
                        
                    !!!
                        bias must be accounted for in the X matrix
                        for simplicity's sake, add the 1 to the first position, and copy all the values over        
                        this must be done for EVERY APPEARANCE OF X

                ...
                
                we also need to account for the bias?
                strip output from each row, replace with 1s?
                
                
                
                TODO TODO
                how do we know when we are done?
                    100 percent recognition???
                    how do we use the starter code
                    
                ----
                lin class - needs to match 74 & 84,
                knn should be 100 on training data, is fuzzy otherwise, just get close
                    
        """
        score_classes = None # todo make this an answer




        # matrix is set up [in,in,...out]
        # so we need to strip out the last element of every row
        # and make that a vector
        # finally
        # we need to do dot product (@) of vector and matrix minus the vector

        #...
        #or is it? 

        # do we do this in place?
        # do we remove the elements??????

        # how do we know when we did it right?????

        scores_classes = np.zeros((row_count,2))



        # Return N x 2 result array
        return scores_classes

    return classifier


def knn_classifier(k, data_matrix):
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix): # todo what is going on with this function? can we reach k?

        # inputs are two var pairs. one feature per row 
        (input_count, _) = input_matrix.shape

        """
            what is the knn algo???
            
            Y^ (x: matrix, k: int) = ( 1/ k ) * SUM FROM (x_i IN N_k(x)) (y_i)
            what does this mean??
            
            what is:
                x_i 
                    row of x?
                N_k(x)
                N
                y_i
                    output?
                    
                    ^^^^ IGNORE ALL OF THIS
            -------
                knn is just a geometric rep of probability, vals associated with 'good' spaces are assumed good
                
                k defs the size of that 'good' space - the 'neighborhood'
                    k = num of neighbors
                        always at least 1, 1 is the self
                        
                the algo is that we calc euclidean dist from the input to all other points in the training data
                
                ^^^^ there is no given algo for this
                    you need to figure it out yourself
                    
                least squares does not relat eto this at all 
                
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

        # REVISE: Always choose class 1
        scores_classes = np.ones((input_count,2))
        

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
