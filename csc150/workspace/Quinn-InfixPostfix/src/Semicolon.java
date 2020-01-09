
/**
 * A token for plus
 * @author xavier
 *
 */
public class Semicolon implements Token
{
	
	private final static int PRECEDENCE=1;
	private String identity;
	
	
	/**
	 * The constructor for this token
	 * 
	 * @param assign what string this will return.
	 * 
	 * This may seem unnescisary but this would let you customize the
	 * format which you get the postfix (+ vs. plus), as well as offer
	 * more modularity for my sorting system.
	 */
	public Semicolon(String assign) {
		identity=assign;
	}
	
	/**
	 * @return a string of the token
	 */
    public String toString() {
    	return identity;
    }
    
    /**
     * Handles the token based on what it is
     * @param s Is the stack the token will handle itself with
     * @returns the next section of string
     */
    public String handle(Stack<Token> s)
    {
    	String toReturn="";
    	
    	while(!s.isEmpty()) {
    		toReturn+=s.pop().toString();
    	}
    	
    	toReturn+=this.toString();
        return toReturn + "\n";
    }
    
    /**
     * @return The precedence of the token
     */
    public int getPrec() {
    	return PRECEDENCE;
    }
}
