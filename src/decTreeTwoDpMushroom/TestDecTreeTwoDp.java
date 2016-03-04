package decTreeTwoDpMushroom;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestDecTreeTwoDp {

	public static int maxDepth;
	public static String epsilon = "1";                       //0.01  0.1  1  10              
	public static int maxiteration = 500;
	public static double equilibriumThreshold = 50;
	                      
	public static void main(String[] args) throws Exception {

		int time = 0;
        double acc[][] = new double[13][300];
        double finalacc[] = new double[13];

        while(time < 300){
        	
        	for(int i=8; i<=20; i++){
    			
    			maxDepth = i;
    	        Random random = new Random();
    			
    			String trainDataPath = "dataset/mushroom.arff";          
    			Instances trainData = null;                              
    			 
    			trainData = (new DataSource(trainDataPath)).getDataSet();       
    			if (trainData.classIndex() == -1)                         
    				trainData.setClassIndex(trainData.numAttributes() - 1);		
    			
    			DecTreeTwoDp tree = new DecTreeTwoDp();
    			
    			tree.setMaxDepth(maxDepth);
    			tree.setEpsilon(epsilon);
    			tree.setMaxIteration(maxiteration);
    			tree.setEquilibriumThreshold(equilibriumThreshold);
    			tree.setSeed(random.nextInt());                      

    			Evaluation eval = new Evaluation(trainData);      

    			eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));  
    			acc[i-8][time] = eval.pctCorrect();
    		}
        	time = time + 1;
        }
       
        for(int i=0; i<13; i++){
        	for(int j=1; j<300; j++){
    	    	acc[i][0] = acc[i][0] + acc[i][j];
    	    }
        }
	    for(int k=0; k<13; k++){
	    	finalacc[k] = (double)(acc[k][0]/300);
	    	System.out.println(finalacc[k]);
	    }
	    
	}
}
