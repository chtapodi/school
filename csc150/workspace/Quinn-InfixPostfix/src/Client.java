
/**
 * This is running an infix to postifx converter on the given input file and
 * printing the output I affirm that I have carried out the attached academic
 * endeavors with full academic honesty, in accordance with the Union College
 * Honor Code and the course syllabus.
 * 
 * @author xavier
 * 
 */
public class Client {

	/**
	 * Just runs the converter
	 */
	public static void main(String[] args) {

		Converter con = new Converter("src/input.txt");

		// Prints the output
		System.out.println(con.convert());

	}

}
