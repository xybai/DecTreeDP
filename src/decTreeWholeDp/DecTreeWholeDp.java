package decTreeWholeDp;

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

public class DecTreeWholeDp extends AbstractClassifier{

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
    private BigDecimal splitbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal splitmarkovbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);
    
    private Random Random = new Random(); 
    public void setSeed(int seed){                          
		Random = new Random(seed);
	}
    
    private int MaxIteration;                            
    public void setMaxIteration(int maxiteration){
    	MaxIteration = maxiteration;
    }
    
    private double EquilibriumThreshold;                 
    public void setEquilibriumThreshold(double equilibriumThreshold){
    	EquilibriumThreshold = equilibriumThreshold;
    }

    private double GiniSen = 2;       
    
	private Node Root;
	
	private Set<Attribute> Attributes = new HashSet<Attribute>();
	private Set<Node> InnerNodes = new HashSet<Node>();

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

	public void buildClassifier(Instances data) throws Exception {
		
		HashSet<Attribute> allAttributes = new HashSet<Attribute>();        
        Enumeration<Attribute> attrEnum = data.enumerateAttributes();  
        while(attrEnum.hasMoreElements()){           
        	allAttributes.add( attrEnum.nextElement() );   
        }
        
        Attributes = new HashSet<Attribute>(allAttributes);

        Root = new Node(data, null, 0);                       
        
        splitbudget = Epsilon.divide(BigDecimal.valueOf(2),MATH_CONTEXT);
        noisebudget = splitbudget;
        splitmarkovbudget = Epsilon.subtract(noisebudget,MATH_CONTEXT);
        
        Set<Attribute> Attrs = new HashSet<Attribute>(Attributes);
        randomSplitTree(Root, Attrs);           
        
        boolean equilibrium = false;
        int iteration = 0;
    
        while(iteration < MaxIteration && !equilibrium){
        	double initialscore = 0;
        	double laterscore = 0;
        	
        	initialscore = gini.score(Root);
        	
        	Node oldnode = (Node)InnerNodes.toArray()[Random.nextInt(InnerNodes.size())];    

        	Node newnode = new Node(oldnode);
        	
        	Attribute a;                      
        	Set<Attribute> subattrs = new HashSet<Attribute>(Attributes);    
        	deleteParAttr(oldnode,subattrs);
        	deleteChiAttr(oldnode,subattrs);            
        	if(subattrs.size()==0){
        		iteration++;
        		continue;
        	}
        	else{
               	a = (Attribute)subattrs.toArray()[Random.nextInt(subattrs.size())];
        	}
        
        	Instances[] Parts = partitionByAttr(newnode.data,a);     
	     	Node[] Children = new Node[a.numValues()];    
	     	
	    	newnode.splitAttr = a;
		    newnode.children = Children;
			
			for(int k=0; k < Parts.length; k++){
				Children[k] = new Node(Parts[k], newnode, k);
				
				makeLeafNode(Children[k]);
			}
		
			int minlen = (oldnode.children.length<newnode.children.length) ? oldnode.children.length : newnode.children.length;
			
			if(minlen==oldnode.children.length){          
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
				if(minlen!=newnode.children.length){
					for(int l=minlen; l<newnode.children.length; l++){
						Set<Attribute> subtreeAttrs = new HashSet<Attribute>(Attributes);
						randomSplitSubtree(newnode.children[l], subtreeAttrs);
					}
				}
			}
			else if(minlen==newnode.children.length){
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
			}
			
			if(oldnode.parent==null){
				newnode.parent = null;
				newnode.index = 0;
				Root = newnode;
			}
			else{
			    Node oldnodepar = oldnode.parent;
				newnode.parent = oldnodepar;
				newnode.index = oldnode.index;
				oldnodepar.children[oldnode.index] = newnode;
			}
		
			laterscore = gini.score(Root);
			
			double initialpro = expProbability(initialscore,splitmarkovbudget);
			double laterpro = expProbability(laterscore,splitmarkovbudget);
				
			double ratio = (double)(laterpro/initialpro);         
			if(ratio>=1){

				removeInnerNodes(oldnode);
				addInnerNodes(newnode);

				double totalscore = gini.score(Root);         
				equilibrium = isEquilibrium(totalscore);          
			}
			else{
				boolean replace = false;
				double randomdouble = Random.nextDouble();
				if(randomdouble<ratio){
					replace = true;
				}
				if(replace){

					removeInnerNodes(oldnode);
					addInnerNodes(newnode);

					double totalscore = gini.score(Root);          
					equilibrium = isEquilibrium(totalscore);          
				}
				else{

					if(oldnode.parent==null){
						Root = oldnode;
					}
					else{
						Node newnodepar = newnode.parent;
						oldnode.parent = newnodepar;
						oldnode.index = newnode.index;
						newnodepar.children[newnode.index] = oldnode;
					}
				}
			}

        	iteration++;                                 
        }

        addNoise(Root, noisebudget);                                 
        
	}

	private void randomSplitTree(Node node, Set<Attribute> Attrs){
		
		Attrs = new HashSet<Attribute>(Attributes);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}
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
	
		makeInnerNode(node);
	
		InnerNodes.add(node);               

		Attribute splitAttr = (Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		
		Instances[] parts = partitionByAttr(node.data, splitAttr);           
		Node[] children = new Node[splitAttr.numValues()];        
		
		node.splitAttr = splitAttr;                       
		node.children = children;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitTree(children[i], Attrs);     
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

	private void deleteChiAttr(Node node, Set<Attribute> Attrs){
	
		if(node.isLeaf==true)
			return;
		Attrs.remove(node.splitAttr);

		for(Node child: node.children){
				
			deleteChiAttr(child,Attrs);		
		}
	}

	private void replaceNode(Node oldnode, Node newnode){
		
		if(oldnode.isLeaf==true){
			makeLeafNode(newnode);
			return;
		}
		
		makeInnerNode(newnode);
		
		Attribute splitAttr = oldnode.splitAttr;
		
		newnode.splitAttr = splitAttr;
		newnode.index = oldnode.index;
		
		Node[] children = new Node[splitAttr.numValues()];
		Instances[] parts = partitionByAttr(newnode.data, splitAttr);
		
		newnode.children = children;
	
		for(int i=0;i<parts.length;i++){
			children[i] = new Node(parts[i],newnode,i);
			
			replaceNode(oldnode.children[i],children[i]);
		}
	}
	
	private void removeInnerNodes(Node node){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			removeInnerNodes(child);
		}
		InnerNodes.remove(node);
	}

	private void addInnerNodes(Node node){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			addInnerNodes(child);
		}
		InnerNodes.add(node);
	}

	private void randomSplitSubtree(Node node, Set<Attribute> Attrs){
		
		Attrs = new HashSet<Attribute>(Attributes);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}
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
	
		makeInnerNode(node);
		
		Attribute splitAttr = (Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		
		Instances[] parts = partitionByAttr(node.data, splitAttr);           
		Node[] children = new Node[splitAttr.numValues()];        
		
		node.splitAttr = splitAttr;                       
		node.children = children;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitSubtree(children[i], Attrs);     
		}
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
    	double beta=bigBeta.doubleValue();                
    	
        double uniform = Random.nextDouble()-0.5;            
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform));  
        
    }

  	private double expProbability(double score, BigDecimal epsilon)        
  	{
  		return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
  	}                  
  	
  	final int BufferSize = 500;   
  	double[] ScoreBuffer = new double[BufferSize];

  	int InitPointer = 0;           
  	int Pointer = 0;               
  	double variance = -1;                 
  	
  	private boolean isEquilibrium(double newScore)
  	{
  		if(InitPointer == BufferSize){
  			variance = Utils.variance(ScoreBuffer);
  			if(variance < EquilibriumThreshold){
  				return true;
  			}
  			InitPointer++;
  		}
  		if( InitPointer < BufferSize ){                 
  			ScoreBuffer[InitPointer++] = newScore;       
  			
  			return false;                                    
  		}
  		if(Pointer==BufferSize)
  			Pointer = 0;
  		ScoreBuffer[Pointer++] = newScore;                                 
  		
  		variance = Utils.variance(ScoreBuffer);          

  		if(variance < EquilibriumThreshold){         
  			InitPointer = 0;
  			Pointer = 0;                                
  			return true; 
  		}
  		return false;                                   
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
