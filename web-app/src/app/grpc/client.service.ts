import { Injectable } from '@angular/core';
import { DroneClient } from '../../protos/DroneServiceClientPb';
import { HelloRequest, HelloReply } from '../../protos/drone_pb';
import * as grpcWeb from 'grpc-web';

export { ClientService, Server };

@Injectable({
  providedIn: 'root'
})
class ClientService {

  server: Server = {
    protocol: 'https://',
    address: 'drone.kandong.dev',
    port: 10000,
    password: 'password'
  };

  client: DroneClient | undefined = undefined;

  constructor() { }

  async SayHello(name: string): Promise<string | null> {

    if (this.client === undefined) {
      return null;
    }

    let request = new HelloRequest();
    request.setName(name);
    let reply = await this.client.sayHello(request, null);
    return reply.getMessage();
  }

  // https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/digest
  async digestMessage(message: string) {
    const msgUint8 = new TextEncoder().encode(message);                           // encode as (utf-8) Uint8Array
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);           // hash the message
    const hashArray = Array.from(new Uint8Array(hashBuffer));                     // convert buffer to byte array
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join(''); // convert bytes to hex string
    return hashHex;
  }

  async connect(server?: Server): Promise<boolean> {
    if (server !== undefined) {
      this.server = server;
    }
    if (this.client !== undefined) {
      // this.client.close()
    }
    let hostname = this.server.protocol + this.server.address + ':' + this.server.port;
    let hash = await this.digestMessage(this.server.password);
    const authInterceptor = new AuthInterceptor(hash)
    const options = {
      unaryInterceptors: [authInterceptor],
      streamInterceptors: [authInterceptor]
    }
    this.client = new DroneClient(hostname, null, options);
    const name = 'world';
    let message = await this.SayHello(name);
    if (message === 'Hello ' + name) {
      return true;
    } else {
      return false;
    }
  }

}


interface Server {
  protocol: string;
  address: string;
  port: number;
  password: string;
}

// https://nicu.dev/posts/typescript-grpc-web-auth-interceptor
class AuthInterceptor {
  token: string

  constructor(token: string) {
    this.token = token
  }

  intercept(request: any, invoker: any) {
    const metadata = request.getMetadata()
    metadata.Authorization = 'Bearer ' + this.token
    return invoker(request)
  }
}

