/**
 * A FileReader object will read tokens from an input file.  The name of
 * the input file is given when the constructor is run.  The lone method,
 * nextToken(), will return the next word in the file as a String.
 * 
 * @author Chris Fernandes
 * @version 2/27/17
 */
import java.io.*;

public class FileReader 
{
    
    private StreamTokenizer st;     //file descriptor

    /**
     * @param fileName path to input file. Can either be a full
     *  path like "C:/project7/proj7_input.txt" or a relative one like
     *  "src/proj7_input.txt" where file searching begins inside
     *  the project folder.
     *  Be sure to use forward slashes to specify folders.  And
     *  don't forget the quotes (it's a String!)
     */
    public FileReader(String fileName) 
    {
        try {
        st = new StreamTokenizer(
                 new BufferedReader(
                     new InputStreamReader(
                         new FileInputStream(fileName))));
        } catch (IOException e) {}
        st.resetSyntax();                     // remove default rules                
        st.ordinaryChars(0, Character.MAX_VALUE); //turn on all chars
        st.wordChars(65,90);                       
        st.wordChars(97,122);       //make the letters into tokens
        st.whitespaceChars(0, 34);
        st.whitespaceChars(36,64);  //make everything else except "#" into
        st.whitespaceChars(91,96);  //whitespace
        st.whitespaceChars(123, Character.MAX_VALUE);
    }

    /**
     * returns the next token as a String. Possible tokens are
     * words, the pound symbol "#" signifying the end of a page,
     * and null which is returned when the end of the file is reached.
     * 
     * @return a word, "#", or null
     */
    public String nextToken() 
    {
        try
        {
        while (st.nextToken() != st.TT_EOF) {
            if (st.ttype < 0)
            {
                return st.sval;
            }
            else
            {
                return String.valueOf((char)st.ttype);
            }
        }
        return null;
        } catch (IOException e) {}
        return "error";
    }
}
