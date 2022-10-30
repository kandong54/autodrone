# Web App

This is a web-based UI to control the drone: [demo](https://io.kand.dev/drone).

Up to now, it has
- top bar component
- login page
- camera page
- not-found page
- gprc service

## Framework & Module

- [Angular](https://angular.io/)
- [Angular Material](https://material.angular.io/)
- [gRPC Web](https://github.com/grpc/grpc-web)

## UI Design
Special thanks to [Angular Components Docs Site](https://github.com/angular/material.angular.io).

## gRPC Web

Wire Format Mode is grpcwebtext to support server streaming calls.

To install Protocol Buffer Compiler on Linux, run [install_protoc.sh](tools/install_protoc.sh).

To compile the proto files:

```shell
npm run proto-windows # proto-linux
```

## Workflow
The [action](/.github/workflows/main.yml) automatically builds and deploys the web to [GitHub Pages](https://pages.github.com/). A [redirct script](https://github.com/kandong54/kandong54.github.io/blob/gh-pages/_layouts/404.html#L6-L21) is required if the web is deployed to a sub directory.
