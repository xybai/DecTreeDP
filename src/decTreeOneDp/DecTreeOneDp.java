package decTreeOneDp;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

public class DecTreeOneDp extends AbstractClassifier{

	private static final long serialVersionUID = 1L;               

    public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN);   
    
	private int MaxDepth;
	public void setMaxDepth(int maxDepth) {
		MaxDepth = maxDepth;
	}
	
	private BigDecimal Epsilon = new BigDecimal(1.0);       
    public void setEpsilon(String epsilon) {                       
	    if (epsilon!=null && epsilon.length()!=0)
	        Epsilon = new BigDecimal(epsilon,MATH_CONTEXT);
    }
    private BigDecimal onelevelbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal splitbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);

    private Random Random = new Random();  
    public void setSeed(int seed){                         
		Random = new Random(seed);
	}

    private double GiniSen = 2;       
    
	private Node Root;
	
	private Set<Attribute> Attributes = new HashSet<Attribute>();

	GiniIndexDirectAdd gini = new GiniIndexDirectAdd();  
	
	public class Node implements Serializable{
		
		private static final long serialVersionUID = 1L;     

		public Instances data;
		
		public int depth;
		
		public Attribute splitAttr;
		
		public Node[] children;
		
		public Node parent;
		
		public int index;
		
		public boolean isLeaf;
		
		public double[] count;             
		public double[] dist;              
		
		public void updateDist(){                      
			dist = count.clone();                        
			if( Utils.sum(dist) != 0.0)                      
				Utils.normalize(dist);           
		}
		
		public Node(Instances data, Node parent, int index){
			
			this.data = data;
			this.parent = parent;
			this.index = index;

			if(parent!=null)
				this.depth = parent.depth + 1;
			else
				this.depth = 1;                  
			
			double[] tempcount = new double[data.numClasses()];                   
			Enumeration<Instance> instEnum = data.enumerateInstances();  
			while(instEnum.hasMoreElements()){                              
				Instance inst = (Instance)instEnum.nextElement();            
				tempcount[(int)inst.classValue()]++;                 
			}
				
			this.count = tempcount;                                   
			this.dist = count.clone();                     
			if(Utils.sum(this.dist) != 0.0)                       
				Utils.normalize(this.dist);              
		}
		
        public Node(Node another){                              
			
			this.data = new Instances(another.data); 
			this.depth = another.depth;                         
			this.isLeaf = another.isLeaf;
			this.count = another.count;
			this.dist = another.dist;
			this.splitAttr = another.splitAttr;
        }
	}

	public void buildClassifier(Instances data) throws Exception {
		
		HashSet<Attribute> allAttributes = new HashSet<Attribute>();   
        Enumeration<Attribute> attrEnum = data.enumerateAttributes();  
        while(attrEnum.hasMoreElements()){        
        	allAttributes.add( attrEnum.nextElement() );      
        }
        
        Attributes = new HashSet<Attribute>(allAttributes);

        Root = new Node(data, null, 0);                      
       
        onelevelbudget = Epsilon.divide(BigDecimal.valueOf(MaxDepth+1),MATH_CONTEXT);    
        
        splitbudget = onelevelbudget;
        noisebudget = onelevelbudget;                        
        
        Set<Attribute> Attrs = new HashSet<Attribute>(Attributes);
        partitionOne(Root, Attrs);              

        addNoise(Root, noisebudget);                               

	}
	
    private void partitionOne(Node node, Set<Attribute> Attrs){

    	Attrs = new HashSet<Attribute>(Attributes);

    	if(node.data.numInstances()==0){
			makeLeafNode(node);
			return;
		}
		
		for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}

		deleteParAttr(node,Attrs);         
		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}

		if(node.depth >= MaxDepth){
			makeLeafNode(node);
			return;
		}
		
		makeInnerNode(node);

		double attrscores[] = new double[Attrs.size()];             
		double expprobabilities[] = new double[Attrs.size()];      
	
		for(int i=0; i<Attrs.size(); i++){
			
			Node tempNode = new Node(node);
			
			Attribute attr1 = (Attribute)Attrs.toArray()[i];        
			tempNode.children = null;
			
			Instances[] tempParts = partitionByAttr(tempNode.data,attr1);
	     	Node[] tempChildren = new Node[attr1.numValues()];           

			for(int k=0; k < tempParts.length; k++){
				tempChildren[k] = new Node(tempParts[k], tempNode, k); 
				tempChildren[k].isLeaf = true;
			}
			tempNode.splitAttr = attr1;
			tempNode.children = tempChildren;
		
	    	attrscores[i] = gini.score(tempNode);
			expprobabilities[i] = expProbability(attrscores[i], splitbudget);         
		}
		
		if(Utils.sum(expprobabilities) != 0){                    
		    Utils.normalize(expprobabilities);         
		}

		for(int j=expprobabilities.length-1; j>=0; j--){
			double sum = 0;
			for(int k=0; k<=j; k++){
				sum += expprobabilities[k];
			}
			expprobabilities[j] = sum;
		}
	
		double randouble = Random.nextDouble();  
        int flag = 0;
        if(randouble < expprobabilities[0]){
        	flag = 0;
        }
        else{
        	 for(int t=0; t<expprobabilities.length-1; t++){
             	if((randouble >= expprobabilities[t])&&(randouble < expprobabilities[t+1])){
             		flag = t+1;
             	}
             }
        }
        
		Attribute goalAttr = (Attribute)Attrs.toArray()[flag];
		
		Instances[] parts = partitionByAttr(node.data, goalAttr);    
	    Node[] children = new Node[goalAttr.numValues()];             

	    node.children = children;              
		node.splitAttr = goalAttr;                        
			
		for(int k=0; k < parts.length; k++){
			children[k] = new Node(parts[k], node, k);   
			
            partitionOne(children[k], Attrs);
		}
	}
		
	private void makeLeafNode(Node node){
		
		node.splitAttr = null;                                          
		node.children  = null;                                    
		node.isLeaf = true;	
	}
	
	private void makeInnerNode(Node node){

		node.isLeaf = false;	
	}

	private Instances[] partitionByAttr(Instances data, Attribute attr){
				
		Instances[] parts = new Instances[attr.numValues()];                  
	    for(int i=0; i<parts.length; i++)
		{
			parts[i] = new Instances( data, data.numInstances() );      
		}
				
		Enumeration<Instance> instEnum = data.enumerateInstances(); 
		while(instEnum.hasMoreElements())                 
		{
			Instance inst = instEnum.nextElement();
			parts[(int)inst.value(attr)].add(inst);          
		}
				
		return parts;                                            
	}
	
	private void deleteParAttr(Node node, Set<Attribute> Attrs){
		
		while(node.parent != null){
			Attrs.remove(node.parent.splitAttr);
			node = node.parent;
		}
		return;
	}

	private void addNoise(Node node, BigDecimal budget){          

    	if(node.isLeaf == true)                                   
    	{ 
    		addNoiseDistribution(node.count, budget);   
    		node.updateDist();                                    
	    	return;
    	}
    	
    	for(Node child : node.children)                         
        {
        	addNoise(child, budget);
        }
    }
   
    private void addNoiseDistribution(double[] count, BigDecimal budget){

    	int maxIndex = Utils.maxIndex(count);                  
    	
    	for(int i=0; i<count.length; i++)
    	{
    		count[i] += laplace(BigDecimal.ONE.divide(budget, MATH_CONTEXT));      
 
    		if(count[i] < 0)	                          
    			count[i] = 0;
    	}
 
    	double sum = Utils.sum(count);                           
    	if(sum <= 0){	                         
    		count[maxIndex] = 1;         
    	}
    } 

    private double laplace(BigDecimal bigBeta){                  
    	
    	double miu = 0.;                                             
    	double beta = bigBeta.doubleValue();                
    	
        double uniform = Random.nextDouble()-0.5;            
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform)); 
        
    }

  	private double expProbability(double score, BigDecimal epsilon)              
  	{
  		return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
  	}                  
  	
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException { 	
    	
    	assert( instance.hasMissingValue() == false);
    	
    	Node node = Root;                                       
    	while(node.isLeaf == false){                     
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return Utils.maxIndex(node.dist);         
    }

	public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {	
		
		assert( instance.hasMissingValue() == false);
		
		Node node = Root;                                 
    	while(node.isLeaf == false){                      
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return node.dist;                          
	}                                       
	
    public String toString() {                    

        return toString(Root);
       
    }

	protected String toString(Node node) {
		
		int level = node.depth;

		StringBuffer text = new StringBuffer();

		if (node.isLeaf == true) {                           

			text.append("  [" + node.data.numInstances() + "]");
			text.append(": ").append(
					node.data.classAttribute().value((int) Utils.maxIndex(node.dist)));
			
			text.append("   Counts  " + distributionToString(node.dist));
			
		} else {                                               
			
			text.append("  [" + node.data.numInstances() + "]");
			for (int j = 0; j < node.children.length; j++) {
				
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				
				text.append(node.splitAttr.name())
					.append(" = ")
					.append(node.splitAttr.value(j));
				
				text.append(toString(node.children[j]));
			}
		}
		return text.toString();
	}

    private String distributionToString(double[] distribution)
    {
           StringBuffer text = new StringBuffer();
           text.append("[");
           for (double d:distribution)
                  text.append(String.format("%.2f", d) + "; ");
           text.append("]");
           return text.toString();             
    }

}
