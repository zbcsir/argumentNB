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
 *  TAN.java
 *    
 *  Author: LiYao (li_yao@bjtu.edu.cn)
 *  Version: 1.0 
 *  Copyright (C) 20 March 2017 LiYao
 *    
 *  NB. This is the TAN of Friedman, Geiger, and Goldszmidt.
 *      "Bayesian Network Classifiers",
 *      Machine Learning, Vol. 29, 131-163, 1996.
 */
package tan;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> Class for building and using a Tree Augmented Naive
 * Bayes (TAN) classifier. This method outperforms naive Bayes, yet at the same
 * time maintains the computational simplicity (no search involved) and
 * robustness that characterizes naive Bayes.<br/>
 * <br/>
 * For more information, see<br/>
 * 
 * <br/>
 * Friedman, N., and Goldszmidt, M. Building Classifiers using Bayesian
 * Networks. In: Proceedings of the Thirteenth National Conference on Artificial
 * Intelligence (AAAI 1996), 1996. pp. 1277 -1284.<br/>
 * 
 * <br/>
 * Friedman, N., Geiger, D., and Goldszmidt, M. Bayesian network classifiers.
 * Machine Learning, Volume 29, Number 2-3, 1997. pp. 131-163. <br/>
 * <p/>
 * <!-- globalinfo-end -->
 *
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
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
 * 
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
 * <!-- technical-bibtex-end -->
 *
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * 
 * <pre>
 *  -R 
 * Choose the root node for building the maximum spanning tree 
 * (default: set by random).
 * </pre>
 * 
 * <pre>
 *  -C 
 * Set the class index location 
 * (default: the last).
 * </pre>
 * 
 * @author LiYao (li_yao@bjtu.edu.cn)
 * @version $Revision: 1.0 $
 * 
 *          Copyright (C) 20 March 2017, LiYao
 * 
 */
public class TAN extends AbstractClassifier
		implements TechnicalInformationHandler {

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
	private int m_ClassIndex = -1;

	/** The counts for each class value. */
	private double[] m_Priors;

	/** Choose the root node for building the maximum spanning tree */
	private static int m_Root;

	/***/
	protected boolean m_debug = true;

	/**
	 * The sums of attribute-class counts. m_CondiPriors[c][k] is the same as
	 * m_CondiCounts[c][k][k]
	 */

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

		// Attributes can handle NOMINAL type properties,
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

	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		// Copy of an instance set
		m_Instances = new Instances(data);

		// Instances number
		m_NumInstances = m_Instances.numInstances();

		// Attribute number
		m_NumAttributes = m_Instances.numAttributes();

		// The data of ClassIndex
		if (m_ClassIndex == -1) {
			m_ClassIndex = data.classIndex();
		}
		// The number of class values
		m_NumClasses = m_Instances.attribute(m_ClassIndex)
				.numValues();

		int index;

		m_NumAttValues = new int[m_Instances.numAttributes() - 1];
		m_StartAttIndex = new int[m_Instances.numAttributes() - 1];
		/** 
		 * The number of values for each attribute. 
		 */
		for (index = 0; index < m_NumAttributes
				&& index != m_ClassIndex; index++) {
			m_NumAttValues[index] = m_Instances.attribute(index)
					.numValues();
		}
		/**
		 * The starting index (in m_CondiCounts matrix) of each attribute.
		 */
		for(index =0 ;index<m_NumAttributes && index !=m_ClassIndex;index++){
			if(index==0)
				m_StartAttIndex[index] = index;
			else{
				for(int j=index;j>0;j--)
					m_StartAttIndex[index] += m_NumAttValues[j-1];
			}
		}

		// The number of values for all attributes, not including class
		index = m_NumAttValues.length - 1;
		while (index >= 0) {
			m_TotalAttValues += m_NumAttValues[index];
			index--;
		}
		// Reserve space
		m_CondiCounts = new long[m_NumClasses][m_TotalAttValues][m_TotalAttValues];
		classins = new int[m_Instances.numClasses()];
		Enumeration<?> enuIns = m_Instances.enumerateInstances();
		while (enuIns.hasMoreElements()) {
			Instance instance = (Instance) enuIns.nextElement();
			/** the number of instance of each class value */
			if(!instance.classIsMissing()){
				processinstance(instance);
//			classins[(int) instance.classValue()]++;
//			for (index = 0; index < m_NumAttributes
//					&& index != m_ClassIndex; index++) {
//				for (attIndex = 0; attIndex < m_NumAttributes
//						&& attIndex != m_ClassIndex; attIndex++) {
//					if(!instance.isMissing(instance.attribute(index))&&!instance.isMissing(instance.attribute(attIndex))){
//					m_CondiCounts[(int) instance
//							.classValue()][(int) instance.value(index)
//									+ m_StartAttIndex[index]][(int) instance
//											.value(attIndex)
//											+ m_StartAttIndex[attIndex]]++;
//				}
//				}
//			}
		}
		}

		// Calculates CMI
		CMI = CalculateCMI();

		// Build a maximum weighted spanning tree
		m_Parents = prim(m_NumAttributes, CMI);
		System.out.println("======m_Parent Test======== ");
		for(int i=0 ;i<m_Parents.length ;i++){
			System.out.print(m_Parents[i] + "\t");
		}
		
	}
	 /**
	   * Handle each instance separately
	   * 
	   * @param Every
	   *          instance of an instances
	   */
	  public void processinstance(Instance ins) {
	    // index of Attribute
	    for (int i = 0; i < m_NumAttributes; i++) {
	      // Excluded class variable
	      if (i != m_ClassIndex) {
	        // index of Attribute
	        for (int j = 0; j < m_NumAttributes; j++) {
	          // Excluded class variable
	          if (j != m_ClassIndex) {
	            m_CondiCounts[(int) ins.classValue()][(int) (m_StartAttIndex[i] + ins
	                .value(i))][(int) (m_StartAttIndex[j] + ins.value(j))]++;

	          }
	        }
	      }

	    }
	    classins[(int) ins.classValue()]++;

	  }

	/**
	 * Calculates conditional mutual information matrix
	 */
	public double[][] CalculateCMI() {
		double CMIArray[][] = new double[m_NumAttributes][m_NumAttributes];
		int classIndex;
		int attIndex1;
		int attValIndex1;
		int attIndex2;
		int attValIndex2;
		// Attribute ai
		for (attIndex1 = 0; attIndex1 < m_NumAttributes; attIndex1++) {
			if (attIndex1 != m_ClassIndex) {
				// Attribute aj
				for (attIndex2 = 0; attIndex2 < m_NumAttributes; attIndex2++) {
					if (attIndex2 != m_ClassIndex
							&& attIndex2 != attIndex1) {
						double singleCMI=0;
						double CMIVal = 0;
						// The value of Attribute ai
						for (attValIndex1 = m_StartAttIndex[attIndex1]; attValIndex1 < m_NumAttValues[attIndex1]
								+ m_StartAttIndex[attIndex1]; attValIndex1++) {
							// The value of Attribute aj
							for (attValIndex2 = m_StartAttIndex[attIndex2]; attValIndex2 < m_NumAttValues[attIndex2]
									+ m_StartAttIndex[attIndex2]; attValIndex2++) {
								// classLabel
								for (classIndex = 0; classIndex < m_NumClasses; classIndex++) {
									// Calculates P(ai,aj,c) Laplace Estimation
									double jointPro = (double) (m_CondiCounts[classIndex][attValIndex1][attValIndex2]
											+ 1)
											/ (m_NumInstances
													+ m_NumClasses
													+ m_NumAttValues[attIndex1]
													+ m_NumAttValues[attIndex2]);
									// Calculates P(ai,aj|c)=(count(ai,aj,C)+1)
									// /(count(c)+Count(ai)+Count(aj))
									double jointCondiPro = (double) (m_CondiCounts[classIndex][attValIndex1][attValIndex2]
											+ 1)
											/ (classins[classIndex]
													+ m_NumAttValues[attIndex1]
													+ m_NumAttValues[attIndex2]);
									// Calculates P(ai|C)=(count(ai,C)+1)
									// /(count(C)+Count(ai))
									double CondiProi = (double) (m_CondiCounts[classIndex][attValIndex1][attValIndex1]
											+ 1)
											/ (classins[classIndex]
													+ m_NumAttValues[attIndex1]);
									// Claculates P(Aj|c)
									double CondiProj = (double) (m_CondiCounts[classIndex][attValIndex2][attValIndex2]
											+ 1)
											/ (classins[classIndex]
													+ m_NumAttValues[attIndex2]);

									singleCMI = jointPro * Math.abs(Math.log(jointCondiPro/ (CondiProi* CondiProj)));

									CMIVal += singleCMI;
								}
							}
						}
						CMIArray[attIndex1][attIndex2] = CMIVal;
					}
				}
			}
		}
		return CMIArray;
	}

	/**
	 * Build a maximum weighted spanning tree
	 * 
	 * @param m_NumAttributes
	 * @param weight
	 * @return the tree
	 */
	public int[] prim(int m_NumAttributes, double[][] weightMatrix) {
		int tree[] = new int[m_NumAttributes];
		double weight[] = new double[m_NumAttributes];
		boolean[] checked = new boolean[m_NumAttributes];
		// 抽取指定的节点
		checked[m_Root] = true;

		// 初始化顶点集合
		for (int i = 0; i < m_NumAttributes; i++) {
			if (i != m_ClassIndex) {
				weight[i] = weightMatrix[m_Root][i];
				tree[i] = m_Root;
				checked[i] = false;
			}
		}
		// 遍历节点集合选择生成树的节点
		for (int i = 0; i < m_NumAttributes; i++) {
			if (i != m_Root) {
				int j = 0;
				// 判定是否抽取该顶点
				for (int k = 0; k < m_NumAttributes; k++) {
					if (k != m_Root) {
						if (weight[k] > EPSILON && (!checked[k])) {
							EPSILON = weight[k];
							j = k;
						}
					}
				}
				// 将顶点加入到新集合中
				checked[j] = true;
				// 根据新加入的顶点，求得最大的权值
				for (int k = 0; k < m_NumAttributes; k++) {
					if (m_Root != k) {
						if ((weightMatrix[j][k] > weight[k])
								&& (!checked[k])) {
							weight[k] = weightMatrix[j][k];
							tree[k] = j;
						}
					}
				}
			}
		}
		// 指定的节点父节点不存在，设为-1;
		tree[m_Root] = -1;
		return tree;
	}

	/**
	 * 
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(2);
		newVector.addElement(new Option(
				"\tChoose the root node for build  the maximum spanning tree\n",
				"R", 1, "-R"));
		newVector.addElement(new Option(
				"\tChoose Estimation Method\n", "E", 1, "-E"));
		return newVector.elements();

	}

	public void setOptions(String[] options) throws Exception {
		String root = Utils.getOption('R', options);
		String classIndex = Utils.getOption("C", options);
		if (root.length() != 0) {
			m_Root = Integer.parseInt(root);
		}
		if (classIndex.length() != 0) {
			m_ClassIndex = Integer.parseInt(classIndex);
		}
		super.setOptions(options);

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

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *                if distribution can't be computed
	 */
	public double[] distributionForInstance(Instance instance)
			throws Exception {

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

	public static void main(String argv[]) {
		try {
			runClassifier(new TAN(), argv);
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}

	}

	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

}
