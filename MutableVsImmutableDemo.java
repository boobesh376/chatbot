public class MutableVsImmutableDemo {
    public static void main(String[] args) {

        System.out.println("===== Student Case =====");

        String student = "Arun";
        System.out.println("Original String: " + student);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(student));

        student = student + " Kumar";  // Creates a new object
        System.out.println("After Concatenation: " + student);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(student));

        StringBuilder sb = new StringBuilder("Arun");
        System.out.println("\nOriginal StringBuilder: " + sb);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(sb));

        sb.append(" Kumar");  // Modifies same object
        System.out.println("After Append (StringBuilder): " + sb);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(sb));

        StringBuffer sbf = new StringBuffer("Arun");
        System.out.println("\nOriginal StringBuffer: " + sbf);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(sbf));

        sbf.append(" Kumar");  // Modifies same object
        System.out.println("After Append (StringBuffer): " + sbf);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(sbf));


        System.out.println("\n===== Library Case =====");

        String book = "Java Basics";
        System.out.println("Original String: " + book);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(book));

        book = "Advanced " + book;  // Creates new object
        System.out.println("After Update: " + book);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(book));

        StringBuilder bookBuilder = new StringBuilder("Java Basics");
        System.out.println("\nOriginal StringBuilder: " + bookBuilder);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(bookBuilder));

        bookBuilder.append(" - 3rd Edition");  // Same object modified
        System.out.println("After Append (StringBuilder): " + bookBuilder);
        System.out.println("Memory reference (hashCode): " + System.identityHashCode(bookBuilder));

        System.out.println("\n=== Code Execution Successful ===");
    }
}
