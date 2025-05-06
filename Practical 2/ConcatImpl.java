import java.rmi.server.UnicastRemoteObject;
import java.rmi.*;

public class ConcatImpl extends UnicastRemoteObject implements Concat {
    ConcatImpl() throws RemoteException {
        super();
    }

    public String join(String s1, String s2) {
        return s1 + s2;
    }
}
