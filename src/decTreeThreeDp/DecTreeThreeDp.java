package decTreeThreeDp;

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

public class DecTreeThreeDp extends AbstractClassifier{

	private static final long serialVersionUID = 1L;           

    public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN);   
    
	private int MaxDepth;
	public void setMaxDepth(int maxDepth) {
		MaxDepth = maxDepth;
	}
	private int temMaxDepth;
	
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
    private int twoMaxIteration = 500;
    
    private double EquilibriumThreshold;              
    public void setEquilibriumThreshold(double equilibriumThreshold){
    	EquilibriumThreshold = equilibriumThreshold;
    }
    private double twoEquilibriumThreshold = 50;
    
    private double GiniSen = 2;       
    
	private Node Root;
	
	private Set<Attribute> Attributes = new HashSet<Attribute>();
	private Set<Node> InnerNodes = new HashSet<Node>();
	
  	final int BufferSize = 75;   
  	double[] ScoreBuffer = new double[BufferSize];
  	                              
  	int InitPointer;            
  	int Pointer;               
  	double variance;                  
  	
	final int twoBufferSize = 50;   
  	double[] twoScoreBuffer = new double[twoBufferSize];
  	                              
  	int twoInitPointer;            
  	int twoPointer;               
  	double twovariance;                 
  	
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
        
        int operatime = ((MaxDepth+2)/3+1);
    
        splitbudget = Epsilon.divide(BigDecimal.valueOf(operatime),MATH_CONTEXT);
        splitmarkovbudget = splitbudget;
        noisebudget = splitbudget;
     
        Set<Attribute> Attrs = new HashSet<Attribute>(Attributes);        
        partitionThree(Root, Attrs);               
        
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
	    	
			expprobabilities[i] = expProbability(attrscores[i], splitmarkovbudget);      
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

			makeLeafNode(children[k]);
		}
	}
 	
  	private void partitionTwo(Node node, Set<Attribute> Attrs){
  		
		temMaxDepth = 0;
		int dep = node.depth;

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
		if(node.depth >= MaxDepth){
			makeLeafNode(node);
			return;
		}
		if(node.depth == MaxDepth-1){
			partitionOne(node,Attrs);
			return;
		}
		
		deleteParAttr(node,Attrs);

		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
		if(Attrs.size()==1){
		
        	Attribute oneattr = (Attribute)Attrs.toArray()[0];
        	
        	Instances[] partss = partitionByAttr(node.data, oneattr);     
	     	Node[] childrenss = new Node[oneattr.numValues()];            

	    	node.children = childrenss;              
			node.splitAttr = oneattr; 
			node.isLeaf = false;
			
			for(int k=0; k < partss.length; k++){
				childrenss[k] = new Node(partss[k], node, k);   
				makeLeafNode(childrenss[k]);
			}
			return;
		}
		
		makeInnerNode(node);
		temMaxDepth = node.depth + 2;

		int tag = 0;
        while(InnerNodes.size()>0){                           
        	InnerNodes.remove((Node)InnerNodes.toArray()[tag]);
        }

        int initialdepth = 2; 
        randomSplitDiffDepTree(node,Attrs,initialdepth);

        boolean equilibrium = false;
        int iteration = 0;
        twoScoreBuffer = new double[twoBufferSize];
	    twoInitPointer = 0;            
		twoPointer = 0;               
		twovariance = -1;                 
		
        while(iteration < twoMaxIteration && !equilibrium){
        	
           	double initialscore = 0;
        	double laterscore = 0;
        	
        	initialscore = gini.score(node);

        	Node oldnode = (Node)InnerNodes.toArray()[Random.nextInt(InnerNodes.size())];   

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
        	
        	Node newnode = new Node(oldnode);
        	
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
        		node = newnode;
        	}
        	else{
        		if(oldnode.depth==dep){
        			node = newnode;
        		}
        		else{
        			Node oldnodepar = oldnode.parent;
    				newnode.parent = oldnodepar;
    				newnode.index = oldnode.index;
    				oldnodepar.children[oldnode.index] = newnode;
        		}
        	}

        	laterscore = gini.score(node);

			double initialpro = expProbability(initialscore,splitmarkovbudget);
			double laterpro = expProbability(laterscore,splitmarkovbudget);
				
			double ratio = (double)(laterpro/initialpro);       
		
			if(ratio>=1){ 
			
				removeInnerNodes(oldnode);
				addInnerNodes(newnode);
				if(oldnode.parent==null){
					Root = newnode;
				}
				else if(oldnode.depth==dep){
					Node oldpar = oldnode.parent;
					node.parent = oldpar;
					node.index = oldnode.index;
					oldpar.children[oldnode.index] = node;
				}
				double totalscore = gini.score(node);         
				equilibrium = twoisEquilibrium(totalscore); 
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
					if(oldnode.parent==null){
						Root = newnode;
					}
					else if(oldnode.depth==dep){
						Node oldpar = oldnode.parent;
						node.parent = oldpar;
						node.index = oldnode.index;
						oldpar.children[oldnode.index] = node;
					}
					double totalscore = gini.score(node);        
					equilibrium = twoisEquilibrium(totalscore); 
				}
				else{
					if(oldnode.parent==null){
						node = oldnode;
					}
					else{
						if(oldnode.depth==dep){
							node = oldnode;
						}
						else{
							Node newnodepar = newnode.parent;
							oldnode.parent = newnodepar;
							oldnode.index = newnode.index;
							newnodepar.children[newnode.index] = oldnode;
						}
					}
				}
			}
        	iteration++;                                 
        }
        
    	for(Node child: node.children){
          	if(child.children==null){
          		continue;
          	}
          	for(Node child2:child.children){
      			partitionTwo(child2,Attrs);
          	}
         }
  	}
	
	private void partitionThree(Node node, Set<Attribute> Attrs){
		
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
		if(node.depth >= MaxDepth){
			makeLeafNode(node);
			return;
		}
		if(node.depth == MaxDepth-1){
			partitionOne(node,Attrs);
			return;
		}
		if(node.depth == MaxDepth-2){
			partitionTwo(node,Attrs);
			return;
		}
		
		deleteParAttr(node,Attrs);

		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
		if(Attrs.size()==1){
		
        	Attribute oneattr = (Attribute)Attrs.toArray()[0];
        	
        	Instances[] partss = partitionByAttr(node.data, oneattr);     
	     	Node[] childrenss = new Node[oneattr.numValues()];            

	    	node.children = childrenss;              
			node.splitAttr = oneattr; 
			node.isLeaf = false;
			
			for(int k=0; k < partss.length; k++){
				childrenss[k] = new Node(partss[k], node, k);   
				makeLeafNode(childrenss[k]);
			}
			return;
		}
		if(Attrs.size()==2){
			partitionTwo(node,Attrs);
			return;
		}
		
		makeInnerNode(node);
		temMaxDepth = node.depth + 3;

		int tag = 0;
        while(InnerNodes.size()>0){
        	InnerNodes.remove((Node)InnerNodes.toArray()[tag]);
        }

        int initialdepth = 1; 
		randomSplitDiffDepTree(node,Attrs,initialdepth);

		boolean equilibrium = false;
        int iteration = 0;
        ScoreBuffer = new double[BufferSize];
	    InitPointer = 0;            
		Pointer = 0;               
		variance = -1;                 
		
        while(iteration < MaxIteration && !equilibrium){
        	
           	double initialscore = 0;
        	double laterscore = 0;
        	
        	initialscore = gini.score(node);

        	Node oldnode = (Node)InnerNodes.toArray()[Random.nextInt(InnerNodes.size())];  

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
        	
        	Node newnode = new Node(oldnode);
        	
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
        		node = newnode;
        	}
        	else{
        		if(oldnode.depth%3==1){
        			node = newnode;
        		}
        		else{
        			Node oldnodepar = oldnode.parent;
    				newnode.parent = oldnodepar;
    				newnode.index = oldnode.index;
    				oldnodepar.children[oldnode.index] = newnode;
        		}
        	}

        	laterscore = gini.score(node);

			double initialpro = expProbability(initialscore,splitmarkovbudget);
			double laterpro = expProbability(laterscore,splitmarkovbudget);
				
			double ratio = (double)(laterpro/initialpro);       
		
			if(ratio>=1){  
			
				removeInnerNodes(oldnode);
				addInnerNodes(newnode);
				if(oldnode.parent==null){
					Root = newnode;
				}
				else if(oldnode.depth%3==1){
					Node oldpar = oldnode.parent;
					node.parent = oldpar;
					node.index = oldnode.index;
					oldpar.children[oldnode.index] = node;
				}
				double totalscore = gini.score(node);       
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
					if(oldnode.parent==null){
						Root = newnode;
					}
					else if(oldnode.depth%3==1){
						Node oldpar = oldnode.parent;
						node.parent = oldpar;
						node.index = oldnode.index;
						oldpar.children[oldnode.index] = node;
					}
					double totalscore = gini.score(node);         
					equilibrium = isEquilibrium(totalscore); 
				}
				else{
					if(oldnode.parent==null){
						node = oldnode;
					}
					else{
						if(oldnode.depth%3==1){
							node = oldnode;
						}
						else{
							Node newnodepar = newnode.parent;
							oldnode.parent = newnodepar;
							oldnode.index = newnode.index;
							newnodepar.children[newnode.index] = oldnode;
						}
					}
				}
			}
        	iteration++;                                
        }
     
    	for(Node child: node.children){
          	if(child.children==null){
          		continue;
          	}
          	for(Node child2:child.children){
          		if(child2.children==null){
              		continue;
              	}
          		for(Node child3:child2.children){
          			partitionThree(child3,Attrs);
          		}
          	}
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
		
	private void randomSplitDiffDepTree(Node node, Set<Attribute> Attrs, int time){
		
		if(time >= 4){
			makeLeafNode(node);
			return;
		}
		
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

			randomSplitDiffDepTree(children[i], Attrs, time+1);     
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
		
		if(node.children==null)
			return;
		for(Node child: node.children){
			addInnerNodes(child);
		}
		InnerNodes.add(node);
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
	
	private void randomSplitSubtree(Node node, Set<Attribute> Attrs){
		
		Attrs = new HashSet<Attribute>(Attributes);

		if(node.depth >= temMaxDepth){                     
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
  	
  	private boolean twoisEquilibrium(double newScore)
  	{
  		if(twoInitPointer == twoBufferSize){
  			twovariance = Utils.variance(twoScoreBuffer);
  			if(twovariance < twoEquilibriumThreshold){
  				return true;
  			}
  			twoInitPointer++;
  		}
  		if( twoInitPointer < twoBufferSize ){                
  			twoScoreBuffer[twoInitPointer++] = newScore;        
  			
  			return false;                                  
  		}
  		if(twoPointer==twoBufferSize)
  			twoPointer = 0;
  		twoScoreBuffer[twoPointer++] = newScore;                             
  		
  		twovariance = Utils.variance(twoScoreBuffer);           

  		if(twovariance < twoEquilibriumThreshold){        
  			twoInitPointer = 0;
  			twoPointer = 0;                               
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
