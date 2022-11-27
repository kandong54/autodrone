import { Injectable } from '@angular/core';
import { DroneClient } from '../../protos/DroneServiceClientPb';
import { HelloRequest, CameraReply, Empty, CameraRequest } from '../../protos/drone_pb';
import { ClientReadableStream } from 'grpc-web';

export { ClientService, Server };

@Injectable({
  providedIn: 'root'
})
class ClientService {

  server: Server = {
    protocol: '',
    address: '',
    port: 0,
    password: '',
    passwordHashed: '',
    passwordChanged: false
  };

  client: DroneClient | null = null;

  constructor() { }

  // https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/digest
  async digestMessage(message: string, salt: string = '3NqlrT9*v8^0') {
    const msgUint8 = new TextEncoder().encode(message + salt);                           // encode as (utf-8) Uint8Array
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);           // hash the message
    const hashArray = Array.from(new Uint8Array(hashBuffer));                     // convert buffer to byte array
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join(''); // convert bytes to hex string
    return hashHex;
  }

  save() {
    localStorage.setItem('server.protocol', this.server.protocol);
    localStorage.setItem('server.address', this.server.address);
    localStorage.setItem('server.port', this.server.port.toString());
    localStorage.setItem('server.passwordHashed', this.server.passwordHashed);
  }
  load() {
    this.server.protocol = localStorage.getItem('server.protocol') ?? 'https://';
    this.server.address = localStorage.getItem('server.address') ?? 'jetson-desktop.local';
    this.server.port = +(localStorage.getItem('server.port') ?? 10000);
    this.server.passwordHashed = localStorage.getItem('server.passwordHashed') ?? '';
  }

  async sayHello(name: string): Promise<string | null> {
    if (this.client === null) return null;
    let request = new HelloRequest();
    request.setName(name);
    let reply = await this.client.sayHello(request, null);
    return reply.getMessage();
  }

  async connect(): Promise<boolean> {
    this.load();
    if (this.server.passwordChanged) {
      this.server.passwordHashed = await this.digestMessage(this.server.password);
      this.server.passwordChanged = false;
    }
    if (this.client !== null) {
      // this.client.close()
    }
    let hostname = this.server.protocol + this.server.address + ':' + this.server.port;
    const authInterceptor = new AuthInterceptor(this.server.passwordHashed)
    const options = {
      unaryInterceptors: [authInterceptor],
      streamInterceptors: [authInterceptor]
    }
    this.client = new DroneClient(hostname, null, options);
    const name = 'world';
    let message = await this.sayHello(name);
    if (message === 'Hello ' + name) {
      this.save();
      return true;
    } else {
      return false;
    }
  }

  getCamera(): ClientReadableStream<CameraReply> | null {
    if (this.client === null) {
      return null;
    }
    let request = new CameraRequest();
    request.setImage(true);
    return this.client.getCamera(request);
  }

  async getImageSize(): Promise<number[] | null> {
    if (this.client === null) return null;
    let reply = await this.client.getImageSize(new Empty(), null);
    return [reply.getWidth(), reply.getHeight(),];
  }

}


interface Server {
  protocol: string;
  address: string;
  port: number;
  password: string;
  passwordHashed: string;
  passwordChanged: boolean;
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

