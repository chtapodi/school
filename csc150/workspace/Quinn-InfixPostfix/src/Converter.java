/**
 * This is an infix- postifx converter that takes a file location with semicolon
 * deliminated equations and converts it to the postfix format.
 * 
 * @author xavier
 * 
 */
public class Converter {
	private FileReader r;

	/**
	 * Constructs a new converter
	 * 
	 * @param location
	 *            is the location of the file to convert
	 */
	public Converter(String location) {
		r = new FileReader(location);
	}

	/**
	 * This runs the conversion on the given text file and returns the converted
	 * string
	 * 
	 * @return Returns the converted string
	 */
	public String convert() {
		String postFix = "";
		String tmp = "";
		Stack pile = new Stack<Token>();
		tmp = r.nextToken();

		while (!tmp.equals("EOF")) { // While next is not EOF
			postFix += analyze(tmp).handle(pile);
			tmp = r.nextToken();
		}

		return postFix;

	}

	/**
	 * Given the input string, makes and returns the correct token
	 * 
	 * @param toConvert
	 * @return
	 */
	private Token analyze(String toConvert) {

		if (toConvert.equals("(")) {
			LeftParen toHandle = new LeftParen(toConvert);
			return toHandle;
		} else if (toConvert.equals(")")) {
			RightParen toHandle = new RightParen(toConvert);
			return toHandle;
		} else if (toConvert.equals("^")) {
			Power toHandle = new Power(toConvert);
			return toHandle;
		} else if (toConvert.equals("*")) {
			Times toHandle = new Times(toConvert);
			return toHandle;
		} else if (toConvert.equals("/")) {
			Divide toHandle = new Divide(toConvert);
			return toHandle;
		} else if (toConvert.equals("+")) {
			Plus toHandle = new Plus(toConvert);
			return toHandle;
		} else if (toConvert.equals("-")) {
			Minus toHandle = new Minus(toConvert);
			return toHandle;
		} else if (toConvert.equals(";")) {
			Semicolon toHandle = new Semicolon(toConvert);
			return toHandle;
		} else {
			Operand toHandle = new Operand(toConvert);
			return toHandle;
		}
	}

}
