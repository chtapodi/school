
/**
 * A token for operands
 * @author xavier
 *
 */
public class Operand implements Token
{
	
	private final static int PRECEDENCE=4;
	private String identity;
	
	
	/**
	 * The constructor for this token
	 * 
	 * @param assign what string this will return.
	 * 
	 * This may seem unnecessary but this would let you customize the
	 * format which you get the postfix (+ vs. plus), as well as offer
	 * more modularity for my sorting system.
	 */
	public Operand(String assign) {
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

        return identity;
    }
    
    /**
     * @return The precedence of the token
     */
    public int getPrec() {
    	return PRECEDENCE;
    }
}
