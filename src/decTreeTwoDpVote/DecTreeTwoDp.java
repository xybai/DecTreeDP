package decTreeTwoDpVote;

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

public class DecTreeTwoDp extends AbstractClassifier{

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
    private BigDecimal splittwobudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);

    private Random Random = new Random();  
    public void setSeed(int seed){                               
		Random = new Random(seed);
	}

    private double GiniSen = 2;       
    
	private Node Root;
	
	private Set<Attribute> Attributes = new HashSet<Attribute>();

	GiniIndexDirectAdd gini = new GiniIndexDirectAdd();
	
	double diffsitscores[];
	double diffsitpros[];
	Attribute allattrs[][];
	int allattrsnum[][];
	int label;
	int id;
	
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
        
        int operatime = (MaxDepth+1)/2+1;

        onelevelbudget = Epsilon.divide(BigDecimal.valueOf(operatime),MATH_CONTEXT);    
        splittwobudget = onelevelbudget;
        noisebudget = onelevelbudget;                        
        
        Set<Attribute> Attrs = new HashSet<Attribute>(Attributes);        
        partitionTwo(Root, Attrs);              
       
        addNoise(Root, noisebudget);                             

	}
	
    private void partitionOne(Node node, Set<Attribute> Attrs){
		
    	Attrs = new HashSet<Attribute>(Attributes);
    	
		deleteParAttr(node,Attrs);
		
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
	    	
			expprobabilities[i] = expProbability(attrscores[i], splittwobudget);       
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
        if(Attrs.size()==1){
        	if(node.depth >= MaxDepth){
        		makeLeafNode(node);
        		return;
        	}
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
        
		if(node.depth >= MaxDepth){
			makeLeafNode(node);
		    return;
		}
		if(node.depth == MaxDepth-1){
			partitionOne(node, Attrs);
			return;
		}
		
		makeInnerNode(node);

	    label = 0;               
		id = 0;                   
		
		int sitnum = 0;               
		for(int i=0; i<Attrs.size(); i++){
			Attribute a = (Attribute)Attrs.toArray()[i];
			sitnum = sitnum + (int)Math.pow(Attrs.size()-1, a.numValues());
		}
		diffsitscores = new double[sitnum];           
		diffsitpros = new double[sitnum];         
		
		allattrs = new Attribute[sitnum][];         
		int v = 0;
		for(int k=0; k<Attrs.size(); k++){
			Attribute b = (Attribute)Attrs.toArray()[k];
			for(int l=v; l<v+(int)Math.pow(Attrs.size()-1, b.numValues()); l++){
				allattrs[l] = new Attribute[b.numValues()+1];		
			}
			v = v + (int)Math.pow(Attrs.size()-1, b.numValues());
		}
		
		allattrsnum = new int[sitnum][];           
		int x = 0;
		for(int k=0; k<Attrs.size(); k++){
			Attribute c = (Attribute)Attrs.toArray()[k];
			for(int l=x; l<x+(int)Math.pow(Attrs.size()-1, c.numValues()); l++){
				allattrsnum[l] = new int[c.numValues()+1];		
			}
			x = x + (int)Math.pow(Attrs.size()-1, c.numValues());
		}
		
		int M[][];                
		M = new int[Attrs.size()][];
		for(int j=0; j<M.length; j++){
			Attribute arrayattr = (Attribute)Attrs.toArray()[j];
			M[j] = new int[arrayattr.numValues()];
		}
		for(int i=0; i<Attrs.size(); i++){
			Attribute arrayattr = (Attribute)Attrs.toArray()[i];
			for(int j=0; j<arrayattr.numValues(); j++){
				M[i][j] = j + 1; 
			}
		}
		
		int selections[] = new int[Attrs.size()];  
		for(int i=0; i<selections.length; i++){
			selections[i] = i;
		}
		
		getallresults(id, selections, M);      
		
		for(int pos=0; pos<allattrsnum.length; pos++){
			
			Node onenode = new Node(node);
			
			int oneline[] = allattrsnum[pos];
			Attribute oneattrs[] = new Attribute[oneline.length];
			for(int q=0; q<oneline.length; q++){
				oneattrs[q] = (Attribute)Attrs.toArray()[oneline[q]];
			}
			allattrs[pos] = oneattrs;
			
			Instances[] parts1 = partitionByAttr(onenode.data, oneattrs[0]);     
	     	Node[] childrens1 = new Node[oneattrs[0].numValues()];             

	    	onenode.children = childrens1;               
			onenode.splitAttr = oneattrs[0];                         

			for(int k=0; k < parts1.length; k++){
				childrens1[k] = new Node(parts1[k], onenode, k);   
				makeInnerNode(childrens1[k]);
			}
			
			int p = 1;
			for(Node child: onenode.children){
				Attribute oneattr = oneattrs[p++];
				
				Instances[] parts2 = partitionByAttr(child.data, oneattr);     
		     	Node[] childrens2 = new Node[oneattr.numValues()];             

		    	child.children = childrens2;               
				child.splitAttr = oneattr;                         
				
				for(int k=0; k < parts2.length; k++){
					childrens2[k] = new Node(parts2[k], child, k);   
                    childrens2[k].isLeaf = true;
				}
			}

			diffsitscores[pos] = gini.score(onenode); 
			diffsitpros[pos] = expProbability(diffsitscores[pos], splittwobudget);     
		}
		
		if(Utils.sum(diffsitpros) != 0){                    
		    Utils.normalize(diffsitpros);           
		}

		for(int j=diffsitpros.length-1; j>=0; j--){
			double sum = 0;
			for(int k=0; k<=j; k++){
				sum += diffsitpros[k];
			}
			diffsitpros[j] = sum;
		}
	
		double randouble = Random.nextDouble();   
        int flag = 0;
        if(randouble < diffsitpros[0]){
        	flag = 0;
        }
        else{
        	 for(int t=0; t<diffsitpros.length-1; t++){
             	if((randouble >= diffsitpros[t])&&(randouble < diffsitpros[t+1])){
             		flag = t+1;
             	}
             }
        }
        
        Attribute finalattrs[] = allattrs[flag];     

		Instances[] parts1 = partitionByAttr(node.data, finalattrs[0]);     
     	Node[] childrens1 = new Node[finalattrs[0].numValues()];            

    	node.children = childrens1;              
		node.splitAttr = finalattrs[0];                      
		
		for(int k=0; k < parts1.length; k++){
			childrens1[k] = new Node(parts1[k], node, k);   
			makeInnerNode(childrens1[k]);
		}
		
		int p = 1;                                
		for(Node child: node.children){
			
			if(child.data.numInstances()==0){       
				makeLeafNode(child);
				p++;                
				continue;
			}
			
			boolean ifsplit = true;            
			for(int i=0; i< child.count.length; i++){        
				if(child.count[i] == child.data.numInstances()){
					makeLeafNode(child);
					ifsplit = false;
				}
			}
			if(ifsplit==false){
				p++;             
				continue;
			}
			
			Attribute oneattr = finalattrs[p++];     
			
			Instances[] parts2 = partitionByAttr(child.data, oneattr);     
	     	Node[] childrens2 = new Node[oneattr.numValues()];             

	    	child.children = childrens2;           
			child.splitAttr = oneattr;                       
			
			for(int k=0; k < parts2.length; k++){
				childrens2[k] = new Node(parts2[k], child, k);   
				partitionTwo(childrens2[k], Attrs);
			}
		}
        
	}

	private void getallresults(int id, int selections[], int M[][]){
		
		String s = "";

		for(int i=0; i<selections.length; i++){

			int t = selections[i];                          
			
			id = iteration(t, M[t], selections, s, id);	
	    }	
		
	}
	
	private int iteration(int pre, int values[], int selections[], String s, int id){
		
		if(values.length>0){
			
			for(int y=0; y<selections.length; y++){
				if(y==pre)
				    continue;
				
				StringBuffer a = new StringBuffer();
				String s1 = a.append(s).append(",").append(pre).append(",").
			        	append(y).toString();
        
		        int[] values1 = new int[values.length-1];
		        System.arraycopy(values, 1, values1, 0, values.length-1);    
	          
				id = iteration(pre,values1,selections,s1,id);
			}
		}
		else{
			id = id + 1;

			int tag = 0;
			int head = 0;
			int tail = 0;
			int first = 0;        
			int last = 0;          
			
			int i = 0;
			while(i < s.length()){
			
				if(i==0){
					int l = s.length()-1;
					while(s.substring(l, l+1).equals(",") == false){
						l--;
					}
					last = l;
					
					i++;
					continue;
				}
				if(i==1){
					head = 0;
					int j = head+1;
					while(s.substring(j, j+1).equals(",") == false){
						j++;
					}
					tail = j;
					allattrsnum[label][tag++] = Integer.parseInt(s.substring(head+1, tail));
					first = Integer.parseInt(s.substring(head+1, tail));
					i = tail;
					head = tail;
					continue;
				}
			
				int j = head+1;
				while(s.substring(j, j+1).equals(",") == false){
					j++;
				}
				tail = j;
				if(tail == last){
					allattrsnum[label][tag++] = Integer.parseInt(s.substring(last+1, s.length()));
					break;
				}
				if(Integer.parseInt(s.substring(head+1, tail)) != first){
					allattrsnum[label][tag++] = Integer.parseInt(s.substring(head+1, tail));
					i = tail;
					head = tail;
				}
				else{
					i = tail;
					head = tail;
				}
				
			}
			label = label + 1;
		}
		return id;	
	
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
    	double beta=bigBeta.doubleValue();               
    	
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
