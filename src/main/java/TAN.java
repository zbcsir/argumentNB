/*
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class TAN extends AbstractClassifier{

	/** for serialization */
	private static final long serialVersionUID = -2224762474573467803L;

	/** The copy of the training instances */
	protected Instances m_Instances;

	/** The number of training instances in the training dataset */
	private double m_NumInstances;
	
	/** The number of training instances with valid class value */
	private double m_NewNumInstances = 0;

	/** The number of attributes, including the class */
	private int m_NumAttributes;

	/** The number of class values */
	protected int m_NumClasses;

	/** The index of the class attribute */
	private int m_ClassIndex;
	
	// The specific class variables below:
	/** The number of values for all attributes, not including class */
	private int m_TotalAttValues;

	/** The starting index (in m_CondiCounts matrix) of each attribute */
	private int [] m_StartAttIndex;

	/** The number of values for each attribute. */
	private int [] m_NumAttValues;

	/** For m_NumClasses * m_TotalAttValues * m_TotalAttValues */
	private long [][][] m_CondiCounts;

	/** The counts for each class value */
	private double [] m_Priors;

	
	/** The matrix of conditional mutual information */
	private double[][] m_CondiMutualInfo;

	/** The array to keep track of an attribute has which parent (Tree) */
	private int[] m_Parents;
	
	/**
	 * Returns a string describing this classifier
	 * 
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "    The class for building and using a very simple Tree "
				+ "Augmented Naive Bayes (TAN)\nclassifier. This method "
				+ "outperforms naive Bayes, yet at the same time maintains the\n"
				+ "computational simplicity (no search involved) and robustness "
				+ "that characterize naive\nBayes. For more information, see\n"
				+ "    [1] Friedman, N. and Goldszmidt, M. (1996). Building "
				+ "classifiers using Bayesian\nnetworks. In: The Proceedings "
				+ "of the National Conference on Artificial Intelligence\n"
				+ "Menlo Park, CA:AAAI Press. PP. 1277-1284. also see\n"
				+ "    [2] "
				+ getTechnicalInformation().toString();
	} // End of globalInfo()
	

	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {

		TechnicalInformation result;
		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR,
				"Friedman, N., Geiger,D. & Goldszmidt, M.");
		result.setValue(Field.YEAR, "1997");
		result.setValue(Field.TITLE, "Bayesian Network Classifiers");
		result.setValue(Field.JOURNAL, "\nMachine Learning");
		result.setValue(Field.VOLUME, "29 ");
		result.setValue(Field.NUMBER, "2-3");
		result.setValue(Field.PAGES, " 131-163");
		return result;
	} // End of getTechnicalInformation()
	

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 * 
	 */
	public Capabilities getCapabilities() {
		// Super is used to call the parent class
		// return an Capabilities method
		Capabilities result = super.getCapabilities();
		// First make it not handle any type of data set
		result.disableAll();

		// Any attribute must be NOMINAL, but it maybe have missing value
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
	  // The class variable must be NOMINAL, but maybe a missing value
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		
		// The least number of instances that can be processed is zero
		result.setMinimumNumberInstances(0);
		
		return result;
	}// End of getCapabilities()
	

	/**
	 * Generates the classifier.
	 * 
	 * @param data
	 *          set of instances serving as training data
	 * @exception Exception
	 *              if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		// Copy of an instance set
		m_Instances = new Instances(data);
		
		// Instances number
		m_NumInstances = m_Instances.numInstances();
		m_NewNumInstances = 0;

		// Attribute number
		m_NumAttributes = m_Instances.numAttributes();

		// Number of classes
		m_NumClasses = m_Instances.numClasses();

		// Check out the class attribute
		if (m_NumClasses < 2)
			throw new Exception("Class values can't less than two!");

		// Class attribute index
		m_ClassIndex = m_Instances.classIndex();
		
		// Any attribute must not be NUMERIC, including the class attribute
		for (int i = 0; i < m_NumAttributes; i++) {
			Attribute attribute = (Attribute) m_Instances.attribute(i);
			if (!attribute.isNominal()) {
					throw new Exception("\nEvery attribute must be nominal. "
							+ "Discretize the dataset with FilteredClassifer please.\n");
			}
		}

		// Judge whether the data set can be processed
		getCapabilities().testWithFail(data);

	 	// the number of attribute values except the class variable
		m_TotalAttValues = 0;
		// Auxiliary array initialization
		m_NumAttValues = new int[(int) m_NumAttributes];
		m_StartAttIndex = new int[(int) m_NumAttributes];

		// the number of instance of each class value
		m_Priors = new double[m_NumClasses];

		// auxiliary array assignment;
		// i is the index of all the attributes of an instance.
		for (int i = 0; i < m_NumAttributes; i++) {
			if (i != m_Instances.classIndex()) {

				m_StartAttIndex[i] = m_TotalAttValues;
				m_NumAttValues[i] = m_Instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i];
			} else {
				m_NumAttValues[i] = m_NumClasses;
			}
		}

		// Debugging 1: Show the auxiliary array assignment
		if (m_Debug) {
			System.out.println("\n\nBuilding A Simple Tree-Augmented Naive Bayes "
					+ "Classifier (Simple TAN1A)");
			System.out.println("The Training Dataset:");
			System.out.println("m_TotalAttValues  = " + m_TotalAttValues);
			System.out.println("m_NumClasses      = " + m_NumClasses);
			System.out.print("m_NumAttValues[]  = ");
			displayVector((int) m_NumAttributes, m_NumAttValues);
			System.out.print("m_StartAttIndex[] = ");
			displayVector((int) m_NumAttributes, m_StartAttIndex);
			System.out.println();
		}

		m_CondiCounts =	new long[m_NumClasses]
				                    [m_TotalAttValues][m_TotalAttValues];
		
		// Take account of the instances one by one
		for (int k = 0; k < m_NumInstances; k++) {

			Instance tempInstance = (Instance) m_Instances.instance(k);

		  // Ignore instance with missing class value
			if (tempInstance.isMissing(m_ClassIndex)) continue;
			
			// Count for all the counters one by one
			addToCounts(tempInstance);
		}

		// Debugging 2: Show the auxiliary array assignments
		if (m_Debug) {
			System.out.println("m_CondiCounts[][][] =");
			// Print out a three-dimensional array
			display3dimensionTable(m_NumClasses, m_TotalAttValues, 
					m_TotalAttValues,	m_CondiCounts);

		}

		// Computing Conditional Mutual Information Matrix
		m_CondiMutualInfo = calulateConditionalMutualInfo();
		
	  // Debugging 3: Show the auxiliary array assignments
		if (m_Debug) {
			System.out.println("Conditional Mutual Information Matrix:");
		  // Print out a two-dimensional array
			displayMatrix((int) m_NumAttributes - 1, (int) m_NumAttributes - 1,
					m_CondiMutualInfo);
		}

		// the beginning node in the Prim algorithm 
		int root = 0;
		
		// Finding the maximum spanning tree
		m_Parents = MaxSpanTree(m_CondiMutualInfo, root, m_ClassIndex);
		System.out.println("======m_Parent Test======== ");
		for(int i=0 ;i<m_Parents.length ;i++){
			System.out.print(m_Parents[i]);
		}
		System.out.println("======Test End========");

	  // Debugging 4: Show the maximum spanning tree
		if (m_Debug) {
			System.out.print("The maximum spanning tree: \nm_Parents[] = ");
		  // Print out an one-dimensional array, including the class 
			displayVector((int) m_NumAttributes, m_Parents);
		}
	}// End of buildClassifier()
	

	/**
	 * Puts an instance values into m_CondiCounts, m_Priors, 
	 * m_NewNumInstances.
	 * 
	 * @param instance
	 *          the instance whose values are to be put into the counts
	 *          variables
	 * @Exception
	 */
	private void addToCounts(Instance instance) throws Exception {

		int tempClassValue = (int) instance.classValue();

		// Add to m_Priors.
		m_Priors[tempClassValue]++;

		// The number of the instances to count
		m_NewNumInstances++;

		// Store this instance's attribute values into an integer array,
		int [] attIndex = new int[m_NumAttributes];

		for (int i = 0; i < m_NumAttributes; i++) {
			if (instance.isMissing(i) || i == m_ClassIndex)
				attIndex[i] = -1;
			else
				attIndex[i] = m_StartAttIndex[i] + (int) instance.value(i);
		} 

		// add to m_CondiCounts
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			// avoid pointless looping
			if (attIndex[att1] == -1) continue;

			for (int att2 = 0; att2 < m_NumAttributes; att2++) {
				if (attIndex[att2] != -1) m_CondiCounts[tempClassValue]
						                   [attIndex[att1]][attIndex[att2]]++;
			}
		}
	} // End of addToCounts()
	

	/**
	 * Calculate conditional mutual information.
	 * 
	 * @throws Exception
	 */
	private double [][] calulateConditionalMutualInfo() throws Exception {

		/** The matrix of conditional mutual information */
		double[][] tempCondiMutualInfo =
				new double[m_NumAttributes][m_NumAttributes];
		
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex) continue;

			// NB. The matrix is symmetric.
			for (int att2 = 0; att2 < att1; att2++) {

				// Condition (att1 == att2) means ignoring the elements 
				// on the diagonal
				if ((att2 == m_ClassIndex) || (att1 == att2)) continue;

			  // adder
				double tempAdder = 0;
						
				// Initialize
				tempCondiMutualInfo[att1][att2] = 0;

				// An item in this matrix
				for (int i = 0; i < m_NumAttValues[att1]; i++) {
					int ai = m_StartAttIndex[att1] + i;

					for (int j = 0; j < m_NumAttValues[att2]; j++) {
						int aj = m_StartAttIndex[att2] + j;
						
						for (int c = 0; c < m_NumClasses; c++) {
							// multiplier
							double tempMultiplier = 1;

							// All Probabilities are estimated using Laplace estimation
							// Compute P(ai,aj|c) -- numerator
							tempMultiplier *= ((double) m_CondiCounts[c][ai][aj] + 1) / 
									    (m_Priors[c] + 
									    		m_NumAttValues[att1] * m_NumAttValues[att2]);

							// Compute P(ai,aj|c)/ P(ai|c),this step:(/ P(ai|c))
							tempMultiplier /= ((double) m_CondiCounts[c][ai][ai] + 1) / 
									    (m_Priors[c] + m_NumAttValues[att1]);

							// Compute P(ai,aj|c)/ ( P(ai|c) * P(aj|c)), i.e. (/ P(aj|c))
							tempMultiplier /= ((double) m_CondiCounts[c][aj][aj] + 1) / 
									    (m_Priors[c] + m_NumAttValues[att2]);

							// ZHW: Testing same as before
							// ZHW: Sometimes it happened, but I do not understand why
							if (tempMultiplier <= 0) {
							  throw new Exception("Bad log m_CondiCounts !");
							}

							// Compute log(P(ai,aj|c)/ ( P(ai|c)*P(aj|c))),this step:log
							tempMultiplier = Math.log(tempMultiplier);
						
							// ZHW: why negtive?
							tempMultiplier = Math.abs(tempMultiplier);

							// Compute P(ai,aj,c)*log(P(ai,aj|c)/( P(ai|c)*P(aj|c)))
							tempMultiplier *= (((double) m_CondiCounts[c][ai][aj] + 1)
										/ (m_NewNumInstances + m_NumClasses * 
												m_NumAttValues[att1] * m_NumAttValues[att2]));
							tempAdder += tempMultiplier;
						}// end of for 5
					}// end of for 4
				}// end of for 3
				
				// Update conditional mutual information
				tempCondiMutualInfo[att1][att2] += tempAdder;
				
				// ZHW (18 August 2017): Symmetric matrix
				tempCondiMutualInfo[att2][att1] += tempAdder;
			} // end of for 2
		} // end of for 1

		return tempCondiMutualInfo;
	} // end of calulateConditionalMutualInfo()
	

	/**
	 * Build the maximum spanning tree, using the Prim algorithm
	 *
	 * @param start
	 *          the root node
	 * @param matrix
	 *          the weights between vertices, a symmetric matrix 
	 * @param index
	 *          the class index
	 * @return an array describing the maximum spanning tree
	 * @throws Exception
	 *    if the start is class attribute, or the matrix is not symmetric 
	 */
	private int [] MaxSpanTree(double[][] matrix, int start, int index)
			throws Exception {

		// The number of vertex
		int numOfVertex = matrix.length;

		// The maximum spanning tree
		int [] tree = new int[numOfVertex];
		
	  // Initialize. "-1" represents no parent for this attribute.
		for (int i = 0; i < numOfVertex; i++) tree[i] = -1;
		
		// check whether the matrix is symmetric or not
		double DELTA = 1.0E-6;
		for (int i = 0; i < numOfVertex; i++)
			for (int j = numOfVertex - 1; j > i; j--)
				if (Math.abs(matrix[i][j] - matrix[j][i]) > DELTA) {
					throw new Exception("The Prim algorithm cannot factor out !");
				}
		
			  	// Start node must not be the class variable
				if (start == index) {
					throw new Exception("The root can't be class attribute!");
				}

		// The node adjVex[j] (belongs to current tree) who has the maximum
		// weight between node j (not belongs to current tree)
		int[] adjVex = new int[numOfVertex];

		// The maximum weight between the node j (not belonging to the current
		// tree so far) and the node adjVex[j] (belonging to the current tree)
		double[] maxWeight = new double[numOfVertex];
		
	  // Add to the current tree
		maxWeight[start] = -1;
		maxWeight[index] = -1;

		
		// Initialize adjVex[] and maxWeight[]
		for (int j = 0; j < numOfVertex; j++) {
			// Finding an edge
			if ((j != index) && (j != start)) {
				adjVex[j] = start;
				maxWeight[j] = matrix[start][j];
			}
		}

		int k = start;

		// Given a connected graph with n vertices, there are (n-1) edges in 
		// its spanning tree. Here, we should not take the class and the start
		// into account  
		for (int i = 0; i < numOfVertex - 2; i++) {
			// Finding a new node k
			for (int j = 0; j < numOfVertex; j++) {
				if ((j != index) && (maxWeight[j] != -1)
						&& (maxWeight[j] > maxWeight[k]))
					k = j;
			}

			// ?
			tree[k] = adjVex[k];

			// ZHW(31 August 2006):
			if (maxWeight[k] < 0)
				throw new Exception("Negative entry in the matrix of "
						+ "conditional mutual information ");
			
			// Update adjVex[] and maxWeight[]
			maxWeight[k] = -1;
			for (int j = 0; j < numOfVertex; j++) {
				if ((j != index) && (maxWeight[j] != -1)
						&& (matrix[k][j] > maxWeight[j])) {
					maxWeight[j] = matrix[k][j];
					adjVex[j] = k;
				}
			}
			// Found an edge
		}
		
		return tree;
	} // End of MaxSpanTree()
	
	
	// There are three printing methods
	// The first is suitable for one-dimensional vector
	public void displayVector(int size, int[] table) {
		for (int i = 0; i < size; i++) {
			System.out.print(table[i]);
			if (i != (size-1)) System.out.print(" & ");
		}
		System.out.println();
	} // END OF THIS METHOD
	

	// The second is suitable for two-dimensional matrix
	public void displayMatrix(int row, int column, double[][] matrix) 
			throws Exception {

		System.out.println("The matrix is below:");
		for (int i = 0; i < row; i++) {
			System.out.print("Row " + i + ": ");

			for (int j = 0; j < column; j++) {
				// Throw over bad entries.
				// if (matrix[i][j] < 0) matrix[i][j] = 0;
				
				// zero item. Why show 5 zeros?
				if (matrix[i][j] < 1.0E-6) {
					System.out.print("0.00000" + " & ");
				} else {
					System.out.print(Utils.doubleToString(matrix[i][j], 5) + " & ");
				}
				
				// Some entries could be more than 1 for using m-estimation
				if (matrix[i][j] > 2) {
					throw new Exception("Bad entry!");
				}
			}
			System.out.println();
		}
		System.out.println("End of the matrix.\n");
	}// END OF THIS METHOD
	

	// The third is suitable for three-dimensional table
	public void display3dimensionTable(int dim1, int dim2, int dim3, 
			 long[][][] table) {
		
		for (int i = 0; i < dim1; i++) {
			System.out.println("i = " + i + ", below:");
			for (int j = 0; j < dim2; j++) {
				for (int k = 0; k < dim3; k++) {
					System.out.print(table[i][j][k]);
					if (k != (dim3-1)) System.out.print(" & ");					
				}
				System.out.println();
			}
			System.out.println();
		}
	}// END OF THIS METHOD
	

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 *
	 * @param instance
	 *          the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *              if there is a problem generating the prediction
	 */
	public double[] distributionForInstance(Instance instance) 
			throws Exception {

		// For debugging
		if (m_Debug) {
			System.out.println(
					"========== Starting distributionForInstance () ==========");
		}

		// Probabilities to be calculated for each class value
		double[] tempProbs = new double[m_NumClasses];

		// Store this instance's attribute values into an integer array,
		int[] index = new int[m_NumAttributes];
		for (int i = 0; i < m_NumAttributes; i++) {
			if (instance.isMissing(i) || i == m_ClassIndex)
				index[i] = -1;
			else
				index[i] = m_StartAttIndex[i] + (int) instance.value(i);
			System.out.println("index[i] = " + index[i]);
		} // end of for

		// Calculate prior probabilities for all possible class values
		for (int c = 0; c < m_NumClasses; c++) {

			// The prior probability using LaPlace estimation
			tempProbs[c] =
						(m_Priors[c] + 1) / 
						(double) (m_NewNumInstances + m_NumClasses);
			
			// Consider effect of each attribute's value
			for (int att = 0; att < m_NumAttributes; att++) {

				if (index[att] == -1)	continue;

				// Determine correct index for the att value in m_CondiCounts
				int aIndex = index[att];

				// Using Laplace estimation
				// The attribute has a parent.
				if ((m_Parents[att] != -1)
								&& (!instance.isMissing(m_Parents[att]))) {
				  // Determine index for parent value in m_CondiCounts
					int pIndex = index[m_Parents[att]];

					// Compute P(c)*P(a|p,c),this step:(*P(a|p,c))
					tempProbs[c] *= ((double) m_CondiCounts[c][pIndex][aIndex] + 1)
							/ (m_CondiCounts[c][pIndex][pIndex] + m_NumAttValues[att]);
				} else {
				  // The attribute doesn't have a parent
					// Compute P(c)*P(a|c),this step:(*P(a|c))
					tempProbs[c] *= ((double) m_CondiCounts[c][aIndex][aIndex] + 1)
											/ (m_Priors[c] + m_NumAttValues[att]);
				}
			} // end of for
		} // end of class for
		
		// For debugging
		if (m_Debug) {
			System.out.println(
					"========== End of distributionForInstance ()   ==========\n");
		}

		return tempProbs;
	} // End of distributionForInstance()
	
	
	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string. 
	 */
	public String toString() {

		StringBuffer text = new StringBuffer();

		text.append("        ---- ");
		text.append("The Simple Tree-Augmented Naive Bayes Classifier ----\n");
		if (m_Instances == null) {
			text.append(": No model built yet.");
		} else {
			try {
				// Print out all the instances one by one 
				text.append("\nThe Instances: \n");
				text.append(
						"  The number of instances in the training instances:  "
								+ (int) m_NumInstances + ".\n");
				text.append(
						"  The number of instances with valid class values:    "
								+ (int) m_NewNumInstances + ".\n");
				
				// Print out string the attribute relationships
				text.append("  The number of attributes (n0t including the class):"
						+ "  " + (m_NumAttributes - 1) + ".\n");
				text.append(
						"\n****************************************************\n");
				for (int i = 0; i < m_NumAttributes; i++) {
					if (m_Parents[i] != -1) {
						text.append(i + ": " + m_Instances.attribute(i).name()
								+ "'s parent is " + m_Parents[i] + "\n");
					} else {
						text.append(i + ": " + m_Instances.attribute(i).name()
								+ " has no parent.\n");
					}
				}
				text.append(
						"****************************************************\n");

			} catch (Exception ex) {
				text.append(ex.getMessage());
			}
		}
		return text.toString();
	} // End of this method.
	
	
	public static void main(String[] argv) {
		
		try {
			//runClassifier(new TAN1A(), argv);
			
			TAN temp = new TAN();
			runClassifier(temp, argv);
			
			System.out.println(temp.globalInfo());
			
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}
}
