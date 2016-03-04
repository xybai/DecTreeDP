package decTreeTwoDpVote;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestDecTreeTwoDp {

	public static int maxDepth;
	public static String epsilon = "1";          //0.01  0.1  1  10                       

	public static void main(String[] args) throws Exception {

		int time = 0;
        double acc[][] = new double[11][300];
        double finalacc[] = new double[11];
          
        while(time < 300){

        	for(int i=4; i<=14; i++){
      			
      			maxDepth = i;
      	        Random random = new Random();
      			
      			String trainDataPath = "dataset/vote.arff";             
      			Instances trainData = null;                               
      			 
      			trainData = (new DataSource(trainDataPath)).getDataSet();       
      			if (trainData.classIndex() == -1)                         
      				trainData.setClassIndex(trainData.numAttributes() - 1);		
      			
      			DecTreeTwoDp tree = new DecTreeTwoDp();
      			
      			tree.setMaxDepth(maxDepth);
      			tree.setEpsilon(epsilon);
      			tree.setSeed(random.nextInt());                     

      			Evaluation eval = new Evaluation(trainData);     

      			eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));  
      			acc[i-4][time] = eval.pctCorrect();
      			
      		}
          	time = time + 1;
        }
        
        for(int i=0; i<11; i++){
          	for(int j=1; j<300; j++){
    	    	acc[i][0] = acc[i][0] + acc[i][j];
            }
        }
  	    for(int k=0; k<11; k++){
  	    	finalacc[k] = (double)(acc[k][0]/300);
  	    	System.out.println (finalacc[k]);
  	    }
  	    
	}
}
