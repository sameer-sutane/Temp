import java.rmi.*;

public class Client {
    public static void main(String[] args) {
        try {
            Concat stub = (Concat) Naming.lookup("rmi://localhost/ConcatService");
            String result = stub.join("Hello ", "World");
            System.out.println("Result from server: " + result);
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
