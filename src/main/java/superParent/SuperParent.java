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

/*
 * 	Author : Zhang Bencai(17120439@bjtu.edu.cn)
 * 	Version: 1.0
 *  2017.9.8 completed
 */
package superParent;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

//check for the current training dataset
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

import java.lang.Math;

public class SuperParent extends AbstractClassifier {
	
	/** for serialization */
	private static final long serialVersionUID = -2224762474573467803L;

	/** The copy of the training instances */
	protected Instances m_Instances;
	
	/** The instances for Leave-One-Out Cross-validation*/
	protected Instances m_InstancesLOO ;

	/** The number of training instances in the training dataset */
	private double m_NumInstances;
	
	/** The number of training instances in the Leave-One-Out 
	 * Cross-validation*/
	private int m_NumInstancesLOO ;
	
	/** The number of training instances with valid class value */
//	private double m_NewNumInstances = 0;

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
	
	/** A m_NumClasses * m_TotalAttValues * m_TotalAttValues array
	 *  for Leave-One-Out */
	private long [][][] m_CountsForLOO ;
	
	/** The counts for each class value */
	private double [] m_Priors;
	private double [] m_PriorsLOO ;
	/** Store an attribute whether an orphan*/
	private boolean[] orphan ;

	/** The array to keep track of an attribute has which parent (Tree) */
	private int[] m_Parents;
	
	/** The number of orphans*/
	private int m_NumOrphan = 0 ;
	
	private int m_NumErrorNB ;
	
	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		Capabilities capabilities =  super.getCapabilities();
		capabilities.disableAll();
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
		capabilities.enable(Capability.DATE_ATTRIBUTES);
		
		capabilities.enable(Capability.NOMINAL_CLASS);
		
		capabilities.setMinimumNumberInstances(0);
		return capabilities ;
	}

	public void buildClassifier(Instances data) throws Exception {
		m_Instances = new Instances(data) ;
		m_NumInstances = m_Instances.numInstances() ;
		m_NumAttributes = m_Instances.numAttributes() ;
		m_NumClasses = m_Instances.numClasses() ;
		// the number of attribute values except the class variable
		m_TotalAttValues = 0 ;
		// Auxiliary array initialization
		m_NumAttValues = new int[m_NumAttributes] ;
		m_StartAttIndex = new int[m_NumAttributes] ;
		// the number of instance of each class value
		m_Priors = new double[m_NumClasses];
		// whether the attribute is an orphan
		orphan = new boolean[m_NumAttributes] ;
		
		if(m_NumClasses < 2){
			throw new Exception("Class values can't less than two!") ;
		}
		
		m_ClassIndex = m_Instances.classIndex() ;
		
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
		// auxiliary array assignment;
		// i is the index of all the attributes of an instance.
		for(int i=0 ;i<m_NumAttributes ;i++){
			if(i != m_ClassIndex){
				m_StartAttIndex[i] = m_TotalAttValues ;
				m_NumAttValues[i] = m_Instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i] ;
			}else{
				m_NumAttValues[i] = m_NumClasses ;
			}
		}
		
		// Debugging 1: Show the auxiliary array assignment
		if (m_Debug) {
			System.out.println("\n\nBuilding A Simple Tree-Augmented Naive Bayes "
					+ "Classifier (Simple TAN1A)");
			System.out.println("The Training Dataset:");
			System.out.println("m_TotalAttValues  = " + m_TotalAttValues);
			System.out.println("m_NumClasses      = " + m_NumClasses);
			System.out.println();
		}
		
		m_CondiCounts = new long[m_NumClasses]
				[m_TotalAttValues][m_TotalAttValues] ;
		
		// Take account of the instances one by one
		for(int i=0 ;i<m_NumInstances ;i++){
			Instance tempIns = m_Instances.instance(i) ;
			// Ignore instance with missing class value
			if(tempIns.isMissing(m_ClassIndex)) continue ;
			addToCount(tempIns);
		}
		m_Parents = init2NaiveBayes() ;
		m_NumErrorNB = leaveOneOut(m_Instances, m_Parents, 1) ;
		for(int i=0 ;i<m_NumAttributes ;i++){
			if((i != m_ClassIndex) && (m_Parents[i] == -1)){
				orphan[i] = true ;
				m_NumOrphan ++ ;
			}
		}
		int superParent = getBestSuperParent() ;
		if(superParent != -1){
			tempAddArcFromSP(superParent) ;
		}
		if(m_Debug){
			printParents(m_Parents);
		}
		
	}
	
	/**
	 * This function is used for Leave-One-Out , which uses N-1(N is 
	 * the number of instances) instances for training and the rest one
	 * is used for classify.This function is used for training an classifier
	 * using the N-1 instances .
	 * 
	 * @param instances  The input data.
	 */
	private void buildClassifierForLOO(Instances instances) {
		m_InstancesLOO = new Instances(instances) ;
		m_NumInstancesLOO = m_InstancesLOO.numInstances() ;
		m_PriorsLOO = new double[m_NumClasses] ;
		m_CountsForLOO = new long[m_NumClasses]
				[m_TotalAttValues][m_TotalAttValues] ;
		// Take account of the instances one by one
		for(int i=0 ;i<m_NumInstancesLOO ;i++){
			Instance tempIns = m_InstancesLOO.instance(i) ;
			// Ignore instance with missing class value
			if(tempIns.isMissing(m_ClassIndex)) continue ;
			addToCountLOO(tempIns);
		}
	}
	
	private void printParents(int[] parent) {
		System.out.print("m_Parent: ");
		for(int i=0 ;i<parent.length ;i++){
			System.out.print(parent[i]+"\t");
		}
		System.out.println();
	}
	
	/**
	 * Puts an instance values into m_CondiCounts, m_Priors, 
	 * m_NewNumInstances.
	 * 
	 * @param instance
	 *    the instance whose values are to be put into the counts
	 *    variables
	 */
	private void addToCount(Instance instance){
		int tempInsClass = (int) instance.classValue() ;
		m_Priors[(int)tempInsClass] ++ ;
		int[] attIndex = attValue(instance) ;
		for(int i=0 ;i<m_NumAttributes ;i++){
			if(attIndex[i] == -1)continue ;
			for(int j=0 ;j<m_NumAttributes ;j++){
				if(attIndex[j] != -1)
					m_CondiCounts[tempInsClass]
							[attIndex[i]][attIndex[j]] ++ ;
			}
		}			
	}
	
	/**
	 * Almost the same with the above function ,but the difference is 
	 * that this function is for Leave-One-Out.
	 * 
	 * @param instance  The instance whose values are to be 
	 * 	put into the counts variables
	 */
	private void addToCountLOO(Instance instance){
		double tempInsClass = instance.classValue() ;
		m_PriorsLOO[(int)tempInsClass] ++ ;
		int[] attIndex = attValue(instance) ;
		for(int i=0 ;i<m_NumAttributes ;i++){
			if(attIndex[i] != -1){
				for(int j=0 ;j<m_NumAttributes ;j++){
					if(attIndex[j] != -1)
						m_CountsForLOO[(int) tempInsClass]
								[attIndex[i]][attIndex[j]] ++ ;
				}
			}
			
		}
	}
	/**
	 * Get the best SuperParent,which is an attribute
	 * @return  The attribute used as the SuperParent
	 * 			that makes the classifier perform best
	 * @throws Exception
	 */
	private int getBestSuperParent() throws Exception {
		int superParent = -1 ;
		int[] parentNB = init2NaiveBayes() ;
		int minError = leaveOneOut(m_Instances ,parentNB ,1) ;
		int[] tmpParent = m_Parents.clone() ;
		if(m_Debug)
			System.out.println("========Get SP Start========");
		for(int i=0 ;i<m_NumAttributes ;i++){
			if((i==m_ClassIndex) || (!orphan[i]))continue ;
			for(int j=0 ;j<m_NumAttributes ;j++){
				if((i!=j) && orphan[j]){
					tmpParent[j] = i ;
				}
			}
			int errorCount = leaveOneOut(m_Instances ,tmpParent ,0) ;
			if(errorCount < minError){
				minError = errorCount ;
				superParent = i ;
			}
		}
		if(m_Debug)
			System.out.println("==========Get SP End============SP is : "+superParent);
		return superParent ;
	}
	
	/**
	 * Make the classifier be a Naive Bayes,
	 * which assign the parent of classIndex to -2
	 * @return  The parent of each attribute
	 */
	private int[] init2NaiveBayes() {
		int[] parent = new int[m_NumAttributes] ;
		for(int i=0 ;i<m_NumAttributes ;i++){
			parent[i] = -1 ;
		}
		return parent ;
	}
	
	/**
	 * Add a temp arc from the SuperParent to evaluate whether the 
	 * classifier is better as the arc is added. 
	 * If the classifier is better,we will add this arc,
	 * if not,the arc will be removed.
	 * 
	 * @param superParent  The superParent we get above.
	 * @return  The parent attribute of each attribute .
	 * @throws Exception
	 */
	private void tempAddArcFromSP(int superParent) throws Exception {
		if(m_Debug)
			System.out.println("errorNB : "+m_NumErrorNB);
		int tempMinError = m_NumErrorNB ;
		boolean isContinue = true ;
		boolean isFirstCycle = true ;
		if(m_Debug)
			System.out.println("=====Temp Add Arc Start======");
		while(isContinue){
			int tmpSP = getBestSuperParent() ;
			if(!isFirstCycle ){
				if(tmpSP != -1)
					superParent = tmpSP ;
			}
			int child = getFavourateChild(superParent, tempMinError) ;
			if(child != -1){
				m_Parents[child] = superParent ;
			}
			if((m_NumOrphan <= 1)||(child == -1)){
				isContinue = false ;
			}
			isFirstCycle = false ;
		}
		if(m_Debug){
			System.out.println("=====Temp Add Arc End======");
			System.out.println();
		}
	}
	
	/**
	 * To find the favorite child of SuperParent
	 * @param superParent
	 * @param tempMinError  The current best classification accuracy.
	 * @return  The favorite child
	 * @throws Exception
	 */
	private int getFavourateChild(int superParent ,int tempMinError) 
			throws Exception{
		int favourateChild = -1 ;
		int[] tempParents = m_Parents.clone() ;
		//To find the best arc
		for(int j=0 ;j<m_NumAttributes ;j++){
			//Every attribute has no more than one parent
			if(!orphan[j] || (j == superParent)) continue ;
			tempParents[j] = superParent ;
			int errorSP = leaveOneOut(m_Instances ,tempParents ,0) ;
			if(errorSP < tempMinError){
				tempMinError = errorSP ;
				favourateChild = j ;
			}
			tempParents[j] = -1 ;
			
		}
		if(favourateChild != -1){
			orphan[favourateChild] = false ;
			m_NumOrphan -- ;
		}
			
		return favourateChild ;
	}
	
	/**
	 * Store this instance's attribute values into an integer array
	 * @param instance  One instance in the instances input
	 * @return  The array of attribute values
	 */
	private int[] attValue(Instance instance) {
		int[] index = new int[m_NumAttributes];
		for (int i = 0; i < m_NumAttributes; i++) {
			if (instance.isMissing(i) || i == m_ClassIndex)
				index[i] = -1;
			else
				index[i] = m_StartAttIndex[i] + (int) instance.value(i);
//			System.out.println("index[i] = " + index[i]);
		}
		return index ;
	}
	
	@Override
	public double[] distributionForInstance(Instance instance)
			throws Exception {
		double tempProp[] = new double[m_NumClasses] ;
		int[] index = attValue(instance) ;
		for(int c=0 ;c<m_NumClasses ;c++){
			tempProp[c] = (m_Priors[c]+1)/(double)
					(m_NumInstances + m_NumClasses);
			for(int att=0 ;att<m_NumAttributes ;att++){
				if(index[att] == -1) continue ;
				int attIndex = index[att] ;
				if((m_Parents[att] != -1) && !instance.isMissing(att)){
					int pIndex = index[m_Parents[att]] ;
					// If the attribute has a parent
					// Compute P(c)*P(a|p,c),this step:(*P(a|p,c))
					tempProp[c] *= ((double)(m_CondiCounts[c][pIndex][attIndex]+1)
							/(m_CondiCounts[c][pIndex][pIndex] + 
									m_NumAttValues[att])) ;
				}else {
					// If the attribute does not have a parent
					// Compute P(c)*P(a|c),this step:(*P(a|c))
					tempProp[c] *= ((double)(m_CondiCounts[c][attIndex][attIndex]
							+1)/(m_Priors[c] + m_NumAttValues[att])) ;
				}
			}
		}
		return tempProp ;
	}
	
	/**
	 * Classify the test instance of Leave-One-Out ,using the classifier that 
	 * is built using the N-1 instances.
	 * @param instance  The test instance of Leave-One-Out
	 * @param parent  The parent of each attribute
	 * @return
	 * @throws Exception
	 */
	public int classifyInstanceSP(Instance instance ,int[] parent) 
			throws Exception {
		double tempProb[] = new double[m_NumClasses] ;
		int[] index = attValue(instance) ;
		for(int c=0 ;c<m_NumClasses ;c++){
			tempProb[c] = (m_PriorsLOO[c]+1)/(double)
					(m_NumInstancesLOO + m_NumClasses) ;
			for(int att=0 ;att<m_NumAttributes ;att++){
				if(index[att] == -1) continue ;
				int attIndex = index[att] ;
				if((parent[att] != -1) && 
						!instance.isMissing(parent[att])){
					int pIndex = index[parent[att]] ;
					// If the attribute has a parent
					// Compute P(c)*P(a|p,c),this step:(*P(a|p,c))
					tempProb[c] *= ((double)(m_CountsForLOO[c]
							[pIndex][attIndex]+1)
							/(m_CountsForLOO[c][pIndex][pIndex] + 
									m_NumAttValues[att])) ;
				}else {
					// If the attribute does not have a parent
					// Compute P(c)*P(a|c),this step:(*P(a|c))
					tempProb[c] *= ((double)(m_CountsForLOO[c]
							[attIndex][attIndex]
							+1)/(m_PriorsLOO[c] + m_NumAttValues[att])) ;
				}
			}
			if(m_Debug)
				System.out.println("prob "+c+" :　"+tempProb[c]);
		}
		double maxProb = tempProb[0] ;
		int classifiedClass = 0 ;
		for(int i=1 ;i<m_NumClasses ;i++){
			if(tempProb[i] > maxProb){
				maxProb = tempProb[i] ;
				classifiedClass = i ;
			}
				
		}
		if(m_Debug){
			System.out.println("maxprob : "+maxProb);
			System.out.println("分类后的类标 : " + classifiedClass);
		}	
		return classifiedClass ;
	}
	
	/**
	 * Evaluate the performance of the SuperParent and the Naive Bayes 
	 * classifier using Leave-One-Out
	 * 
	 * @param data  The data used for training classifier
	 * @param parent  The parent node for every attribute after trying
	 * 					to add an arc
	 * @param flag  0 represent the Leave-One-Out of SuperParent , and
	 *				1 represent the Leave-One-Out of Naive Bayes 				
	 * @return  The number of misclassification
	 * @throws Exception
	 */
	private int leaveOneOut(Instances data ,int[] parent ,int flag) 
			throws Exception {
//		double[] testPro = new double[m_NumClasses];
		int num = (int) data.numInstances() ;
		int lebelClassify ;
		int sumError = 0 ;
		if(flag == 0){
			if(m_Debug)
				System.out.println("========SP LOO Start========");
			for(int i=0 ;i<num ;i++){
				Instance tmpIns = data.get(i) ;
				int insClass = (int)tmpIns.classValue() ;
				data.remove(i) ;
				buildClassifierForLOO(data);
				lebelClassify = classifyInstanceSP(tmpIns ,parent) ;
				if(m_Debug)
					System.out.println("原始数据类别 :　"+insClass);
	//			System.out.println();
				if(Math.abs(lebelClassify - insClass) > 0){
					sumError ++ ;
				}
				data.add(i, tmpIns);
			}
			if(m_Debug)
				System.out.println("========SP LOO End========");
		}else if(flag == 1){
			if(m_Debug)
				System.out.println("========NB LOO Start========");
			for(int i=0 ;i<num ;i++){
				Instance tmpIns = data.get(i) ;
				int insClass = (int)tmpIns.classValue() ;
				data.remove(i) ;
				buildClassifierForLOO(data);
				lebelClassify = classifyInstanceSP(tmpIns ,parent) ;
				if(Math.abs(lebelClassify - insClass) > 0){
					sumError ++ ;			
				}
				data.add(i, tmpIns);
			}
			if(m_Debug){
				System.out.println();
				System.out.println("========NB LOO End========");
			}
			
		}
		if(m_Debug){
			System.out.println("错误数："+sumError);
			System.out.println();
		}
		
		return sumError ;
	}
	
	public static void main(String[] args) {	
		runClassifier(new SuperParent(), args);
	}
}
