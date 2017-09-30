/*
 *  This program is free software; you can redistribute it and/or 
 *  modify it under the terms of the GNU General Public License as 
 *  published by the Free Software Foundation; either version 2 of 
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *  TAN2A.java
 *    
 *  Author: Zhang Bencai (17120439@bjtu.edu.cn)
 *  Version: 2.1.0 
 *  Copyright (C) 21 August 2017 Zhang Bencai
 *    
 *  NB. This is the TAN of Friedman, Geiger, and Goldszmidt.
 *      "Bayesian Network Classifiers",
 *      Machine Learning, Vol. 29, 131-163, 1997.
 */

package tan;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 <!-- globalinfo-start -->
 * Class for building and using a Tree Augmented Naive Bayes (TAN) 
 * classifier. This method outperforms naive Bayes, yet at the same 
 * time maintains the computational simplicity (no search involved) and
 * robustness that characterizes naive Bayes.<br/> 
 * <br/>
 * For more information, see<br/>
 * 
 * <br/>
 * Friedman, N., and Goldszmidt, M. Building Classifiers using 
 * Bayesian Networks. In: Proceedings of the Thirteenth National 
 * Conference on Artificial Intelligence (AAAI 1996), 1996. pp. 1277
 * -1284.<br/>
 * 
 * <br/>
 * Friedman, N., Geiger, D., and Goldszmidt, M. Bayesian network 
 * classifiers. Machine Learning, Volume 29, Number 2-3, 1997. 
 * pp. 131-163. <br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Friedman1996,
 *    author = {Friedman, N. and Goldszmidt, M.},
 *    booktitle = {Proceedings of the Thirteenth National Conference on 
 *                 Artificial Intelligence (AAAI 1996)},
 *    pages = {1277-1284},
 *    series = {AAAI 1996},
 *    title = {Building Classifiers using Bayesian Networks},
 *    publisher = {Menlo Park, CA: AAAI Press},
 *    year = {1996}
 * }
 * </pre>
 * 
 * BibTeX:
 * <pre>
 * &#64;article{Friedman1997,
 *    author = {Friedman, N., Geiger, D., and Goldszmidt, M.},
 *    journal = {Machine Learning},
 *    number = {2-3},
 *    pages = {131-163},
 *    title = {Bayesian network classifiers},
 *    volume = {29},
 *    year = {1997}
 * }    
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *  
 * <pre> -M 
 * If set, delete all the instances with any missing value 
 * (default: false).</pre>
 * 
 * <pre> -R 
 * Choose the root node for building the maximum spanning tree 
 * (default: set by random).</pre>
 * 
 *  * <pre>
 *  -C 
 * Set the class index location 
 * (default: the last). </pre>
 * 
 * @author Zhang Bencai (17120439@bjtu.edu.cn)
 * @version : 1.0.0 $
 * 
 * Copyright (C) 12 August 2011, Zhihai Wang
 * 
 */

/**
 * 21 August 2017, Zhang Bencai: All classifiers extends AbstractClassifier.java, and
 * Classifier.java implements the following five interfaces: (1) Cloneable, (2)
 * Serializable, (3) OptionHandler, (4) CapabilitiesHandler, (5)
 * RevisionHandler.
 * 
 */
public class TAN2AA extends AbstractClassifier implements OptionHandler,
  WeightedInstancesHandler, UpdateableClassifier, TechnicalInformationHandler {

  /** for serialization */
  private static final long serialVersionUID = -2224762474573467803L;

  /** The copy of the training instances */
  protected Instances m_Instances;

  /** The number of training instances in the training instances. */
  private double m_NumInstances;

  /**
   * The number of training instances with valid class value observed in
   * dataset.
   */
  private double m_SumInstances = 0;

  /** The number of attributes, including the class. */
  private int m_NumAttributes;

  /** The number of class values. */
  protected int m_NumClasses;

  /** The index of the class attribute. */
  private int m_ClassIndex;

  /** The counts for each class value. */
  private double[] m_Priors;

  /** Choose the root node for building the maximum spanning tree */
  private int m_Root;

  /***/
  protected boolean m_debug = true;

  /**
   * The sums of attribute-class counts. m_CondiPriors[c][k] is the same as
   * m_CondiCounts[c][k][k]
   */
  // ZHW (12 August 2011):
  // private long [][] m_CondiPriors;

  /** For m_NumClasses * m_TotalAttValues * m_TotalAttValues. */
  private long[][][] m_CondiCounts;

  /** The number of values for all attributes, not including class. */
  private int m_TotalAttValues;

  /** The starting index (in m_CondiCounts matrix) of each attribute. */
  private int[] m_StartAttIndex;

  /** The number of values for each attribute. */
  private int[] m_NumAttValues;

  /**
   * The counts (frequency) of each attribute value for the dataset. Here for
   * security, but it can be used for weighting.
   */

  /** Count for P(ai, aj). Used in M-estimation */
  private int[][] AandB;

  /** The Smoothing parameter for M-estimation */
  private final double SMOOTHING = 5.0;

  /** The matrix of conditional mutual information */
  private double[][] CMI;

  /** The minimum item in the matrix of conditional mutual information */
  private double EPSILON = 1.0E-4;

  /** The array to keep track of which attribute has which parent. (Tree) */
  private int[] m_Parents;

  /** the number of instance of each class value */
  private int[] classins;
  
  /**The estimate mode. 0 represent Laplace Estimation. 
   * 1 represent M-estimation*/
  private int estimateMode = 0;

  /**
   * Returns a string describing this classifier
   * 
   * @return a description of the classifier suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {

    return "Class for building and using a Tree Augmented Naive Bayes"
        + "(TAN) classifier.This method outperforms naive Bayes, yet "
        + "at the same "
        + "time maintains the computational simplicity(no search involved) "
        + "and robustness that characterize naive Bayes.\n\n"
        + "For more information, see\n\n"
        + "Friedman, N. & Goldszmidt, M. (1996). Building classifiers using "
        + "Bayesian networks. In: The Proceedings of the National Conference "
        + "on Artificial Intelligence(pp.1277-1284).Menlo Park, CA:AAAI Press."
        + "also see \n\n  Friedman, N., Geiger,D. & Goldszmidt, M. (1997). "
        + "Bayesian Network Classifiers. Machine Learning, Vol.29,pp.131-163";
  } // End of globalInfo()

  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    // Super is used to call the parent class
    // return an Capabilities method
    Capabilities result = super.getCapabilities();
    // First make it not handle any type of data set
    result.disableAll();

    // Attributes锛欳an handle NOMINAL type properties,
    // properties can have default values
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // A class variable can be a NOMINAL
    // a class variable can have a default value.
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // The least number of instances that can be processed is zero
    result.setMinimumNumberInstances(0);

    return result;
  }

  /**
   * Generates the classifier.
   * 
   * @param instances
   *          set of instances serving as training data
   * @exception Exception
   *              if the classifier has not been generated successfully
   */
  public void buildClassifier(Instances data) throws Exception {
    // TODO Auto-generated method stub
    // Judge whether the data set can be processed
    getCapabilities().testWithFail(data);

    // Copy of an instance set
    m_Instances = new Instances(data);

    // Instances number
    m_NumInstances = m_Instances.numInstances();

    // Attribute number
    m_NumAttributes = m_Instances.numAttributes();

    // Number of classes
    m_NumClasses = m_Instances.numClasses();
    // Class attribute index
    m_ClassIndex = m_Instances.classIndex();
    // the number of attribute values except the class variable
    m_TotalAttValues = 0;
    // Auxiliary array initialization
    m_NumAttValues = new int[(int) m_NumAttributes];
    m_StartAttIndex = new int[(int) m_NumAttributes];
    // the number of instance of each class value
    classins = new int[m_NumClasses];
    // Pretreatment
    if (m_NumClasses < 2)
      throw new Exception("Class values can't less than two!");
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

    if (m_debug) {
    	System.out.println("m_TotalAttValues" + m_TotalAttValues);
    	System.out.println("m_NumClasses: " + m_NumClasses);
    	System.out.println();
    	System.out.println("Auxiliary array output:m_NumAttValues");
    	onedimenout((int) m_NumAttributes, m_NumAttValues);
    	System.out.println("Auxiliary array output:m_StartAttIndex");
    	onedimenout((int) m_NumAttributes, m_StartAttIndex);

    }
    m_CondiCounts = new long[m_NumClasses][m_TotalAttValues][m_TotalAttValues];
    
    // Handle each instance separately
    for (int i = 0; i < m_NumInstances; i++) {
    	processinstance(m_Instances.instance(i));
    }
//    if (m_debug) {
//    	System.out.println("array output:m_CondiCounts锛�");
//    	threedimout(m_NumClasses, m_TotalAttValues, m_TotalAttValues,
//          m_CondiCounts);
//
//    }
    
    // Computing conditional mutual information matrix
    CMI = new double[(int) m_NumAttributes - 1][(int) m_NumAttributes - 1];
    CMI = computeCMI();
    
    if (m_debug) {
    	System.out.println("Conditional mutual information matrix output");
    	twodimout((int) m_NumAttributes - 1, (int) m_NumAttributes - 1, CMI);
    }

    // Generating a one-dimensional array, get the tree
    m_Parents = prim((int) m_NumAttributes, CMI);
    if (m_debug) {
    	System.out.println("Tree output:m_Parents");
    	onedimenout((int) m_NumAttributes, m_Parents);
    }
  }

  /**
   * make the instance map to m_CondiCounts
   * @param instance
   */
  private void processinstance(Instance instance) {
	// TODO Auto-generated method stub
	  for(int i=0 ;i<m_NumAttributes ;i++){
		  if(i != m_ClassIndex){
			  for(int j=0 ;j<m_NumAttributes ;j++){
				  if(j != m_ClassIndex){
					  //m_CondiCounts对应位置上加1
					  m_CondiCounts[(int)instance.classValue()]
						[(int)(m_StartAttIndex[i] + instance.value(i))]
						[(int)(m_StartAttIndex[j] + instance.value(j))] ++ ;
				  }
			  }
		  }
	  }
	  classins[(int)instance.classValue()] ++ ;
  }
	  

  /**
   * Computing mutual information matrix
   * @return the CMI between each pair nodes
   */
  public double[][] computeCMI() {
	  double[][] attrCMI = new double[m_NumAttributes][m_NumAttributes] ;
	  int attri ;
	  int attrj ;
	  int attrvi ;
	  int attrvj ;
	  int classv ;
	  for(attri=0 ;attri<m_NumAttributes ;attri++){
		  //Attribute i
		if(attri != m_ClassIndex){
		  for(attrj=0 ;attrj<m_NumAttributes ;attrj++){
			//Attribute j
				  
			if(attrj != m_ClassIndex && attri != attrj){
			  double singlePairCMI = 0 ;
			  double attrValSumCMI = 0 ;
			  for(attrvi = m_StartAttIndex[attri] ;attrvi < 
					m_StartAttIndex[attri] + m_NumAttValues[attri] ;attrvi++){
			    for(attrvj = m_StartAttIndex[attrj] ;attrvj < 
			    	m_StartAttIndex[attrj] + m_NumAttValues[attrj] ;attrvj++){
				  for(classv=0 ; classv<m_NumClasses ;classv++){
								  
					// Calculates P(ai,aj,c) using Laplace Estimation
					double jointPro = (double) (m_CondiCounts[classv][attrvi]
								[attrvj]+ 1)/ (m_NumInstances
								+ m_NumClasses
								+ m_NumAttValues[attri]
								+ m_NumAttValues[attrj]);
								  
				    // Calculates P(ai,aj|c)=(count(ai,aj,C)+1)
				    // /(count(c)+Count(ai)+Count(aj))
				    double jointConditionPro = (double)(m_CondiCounts[classv]
				    		[attrvi][attrvj]+1)/(classins[classv]
							    + m_NumAttValues[attri]
							    + m_NumAttValues[attrj] ) ;
				  
				    // Calculates P(ai|C)=(count(ai,C)+1)
				    // /(count(C)+Count(ai))
				    double conditionPro_i = (double)(m_CondiCounts[classv]
						  [attrvi][attrvi]+1)/(classins[classv]
							+m_NumAttValues[attri]) ;
					// Claculates P(Aj|c)=(count(ai,C)+1)/(count(C)+count(aj))
				    double conditionPro_j = (double)(m_CondiCounts[classv]
						  [attrvj][attrvj]+1)/(classins[classv]
							+m_NumAttValues[attrj]) ;
								  
				    singlePairCMI = jointPro * Math.log(jointConditionPro
						  / (conditionPro_i* conditionPro_j));
				    attrValSumCMI += singlePairCMI ;
				  }
				}
					  }
				attrCMI[attri][attrj] = attrValSumCMI ;
			}
		  }
	    }		  
	  }
	return attrCMI;
  
  }

  /**
   * prim return a maximum weighted spanning tree of this graph
   * @param numAttributes : The number of attributes, 
   * 	including the class.
   * @param weight : the CMI value between two nodes
   * @return the maximum weighted spanning tree of this graph
   */
  public int[] prim(int numAttributes, double[][] weight) {
	  //存放节点的父节点
	  int[] tree = new int[numAttributes];
	  //存放节点到根节点的权重
	  double[] weight_set = new double[numAttributes] ;
	  //是否遍历
	  boolean[] visited = new boolean[numAttributes] ;
	  visited[m_Root] = true ;
	  for(int i=0 ;i<numAttributes ;i++){
		  if(i!= m_ClassIndex && i != m_Root){
			  weight_set[i] = weight[m_Root][i] ;
			  tree[i] = m_Root ;
			  visited[i] = false ;
		  }
	  }
	  
	  for(int i=0 ;i<numAttributes ;i++){
		  if(i != m_Root){
			  int node = 0 ;
			  //找出到根节点CMI最大的节点，将其index赋值给node
			  for(int j=0 ;j<numAttributes ;j++){
				  if(j != m_Root && weight_set[i] > EPSILON && !visited[i]){
						  EPSILON = weight_set[i] ;
						  node = j ;
				  }
				  
			  }
			  visited[node] = true ;
			  //根据新加入的节点，找出最大权重
			  for(int k=0 ;k<numAttributes ;k++){
				  if(k != m_Root && weight[node][k] > weight_set[k] && (!visited[k])){
					  weight_set[k] = weight[node][k] ;
					  tree[k] = node ;
				  }
			  }
			  
		  }
	  }
	  //根节点无父节点，将其父节点设为-1
	  tree[m_Root] = -1 ;
	  return tree;
  }
  
  private double mEstimate(int x) {
	double theat = 0 ;
	double alfa = m_NumInstances * (m_Priors[x])/
			(m_NumInstances*m_Priors[x] + SMOOTHING);
	return theat ;
	
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   * 
   * @param instance
   *          the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception
   *              if there is a problem generating the prediction
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
	  double[] probs = new double[m_NumClasses];
		int index = 0;
		// Calculates P(Ci|X)~P(Ci)*P(X|Ci)
		// ,P(Ci)=(Count(Ci)+1)/(m_NumInstance+m_NumClass)
		for (index = 0; index < m_NumClasses; index++) {
			double ClassPro = (double)(classins[index] )
					/ (m_NumInstances + m_NumClasses);
			int attIndex;
			// Calculates P(X|C)
			double result = 1;
			for (attIndex = 0; attIndex < m_NumAttributes; attIndex++) {
				if (attIndex != m_ClassIndex) {
					// Calculates P(Ak|C)
					if (m_Parents[attIndex] == -1) {
						double part1 =(double) (m_CondiCounts[index][(int) instance
								.value(attIndex)
								+ m_StartAttIndex[attIndex]][(int) instance
										.value(attIndex)
										+ m_StartAttIndex[attIndex]]);
						double part2 =(double) classins[index]+m_NumAttValues[attIndex];
						
						double part = part1/part2;
						
						result *= part;
					}
					// Calculates
					// P(Ai|Aj,Ci)=(Count(Ai,Aj,Ci)+1)/(Count(Aj,C)+Count(Ai))  
					else {
						int parentIndex = m_Parents[attIndex];
						double part1 =(double) m_CondiCounts[index][(int) instance
								.value(attIndex)
								+ m_StartAttIndex[attIndex]][(int) instance
										.value(parentIndex)
										+ m_StartAttIndex[parentIndex]]
								+ 1;
						double part2 =(double) m_CondiCounts[index][(int) instance
								.value(parentIndex)
								+ m_StartAttIndex[parentIndex]][(int) instance
										.value(parentIndex)
										+ m_StartAttIndex[parentIndex]]
								+ m_NumAttValues[attIndex];
						
						double part = part1 / part2;
						
						result *= part;
					}
				}
			}
			result *= ClassPro;
			probs[index] = result;
		}
		return probs;

  }
 
  //output the one dimen array
  private void onedimenout(int length, int[] data) {
		// TODO Auto-generated method stub
	  for(int i=0 ;i<length ;i++){
		  System.out.print(data[i] + "\t");
	  }
		
  }

  
  // output the matrix
  private void twodimout(int dim1, int dim2, double[][] cmi) {
	// TODO Auto-generated method stub
	  for(int i=0 ;i<dim1 ;i++){
		  if(i != m_ClassIndex){
			  for(int j=0 ;j<dim2 ;j++){
				  if(j != m_ClassIndex){
					  new DecimalFormat().format(1.234) ;
					  System.out.print(cmi[i][j] + "\t");
				  }
					  
			  }
			  System.out.println();
		  }
	  }
  }

  // Output of three-dimensional array
  public void threedimout(int dim1, int dim2, int dim3, long[][][] line) {
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          System.out.print(line[i][j][k] + "\t");

        }
        System.out.println();
      }
      System.out.println();
    }

  }

  /***/
  @SuppressWarnings({ "unchecked", "rawtypes" })
  public Enumeration listOptions() {
    Vector newVector = new Vector(2);
    newVector.addElement(new Option(
        "\tChoose the root node for build  the maximum " + "spanning tree\n",
        "R", 1, "-R"));
    newVector.addElement(new Option("\tChoose Estimation Method\n", "E", 1,
        "-E"));
    return newVector.elements();

  }

  public void setOptions(String[] options) throws Exception {
    String root = Utils.getOption('R', options);
    if (root.length() != 0) {
      m_Root = Integer.parseInt(root);

    }

    super.setOptions(options);

    // Utils.checkForRemainingOptions(options);
  }

  public String[] getOptions() {
    String[] superOptions = super.getOptions();
    String[] options = new String[superOptions.length + 5];
    int current = 0;

    options[current++] = "-R";
    options[current++] = "" + m_Root;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  public static void main(String[] argv) {
    try {
      runClassifier(new TAN2AA(), argv);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }

  }

  public TechnicalInformation getTechnicalInformation() {
	// TODO Auto-generated method stub
	return null;
  }

  public void updateClassifier(Instance instance) throws Exception {
	// TODO Auto-generated method stub
	
  }

}
